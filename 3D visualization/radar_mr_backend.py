import numpy as np
import librosa
import pyaudio
import onnxruntime as ort
import time
import warnings
import asyncio
import websockets
import json
import math

warnings.filterwarnings('ignore')

# Basic hardware configuration
RATE = 16000
CHANNELS = 4  # Strictly use 4 channels (compatible with UAC1.0)
CHUNK = 1024

TARGET_HORN_ID = 0
TARGET_SIREN_ID = 1

BUFFER_SIZE = 51200
PROCESS_INTERVAL = 8000
DISTANCE_WINDOW = 16000

SIREN_NEAR_THRESH = 1.0
SIREN_FAR_THRESH = 4.0
HORN_NEAR_THRESH = 0.010
HORN_FAR_THRESH = 0.032


# Radar core algorithms (distance estimation & direction finding)
def gcc_phat(sig1, sig2, max_tau=10):
    """
    Computes GCC-PHAT cross-correlation bounded within the hardware array's
    maximum possible physical microphone propagation delay time.
    """
    n = len(sig1) + len(sig2) - 1
    N = 2 ** (int(np.log2(n)) + 1)
    X1 = np.fft.fft(sig1, N)
    X2 = np.fft.fft(sig2, N)
    cross_correlation = X1 * np.conj(X2)
    phat_weighting = np.abs(cross_correlation)
    phat_weighting[phat_weighting == 0] = 1e-10
    cc = np.fft.ifft(cross_correlation / phat_weighting).real

    # CRITICAL FIX: Extract only the physically valid microsecond delay window.
    # For a circular array, sound travel limits delay between oppositional mics to under ~5 samples.
    # cc[0 to max_tau] handles positive delays; cc[-max_tau to -1] handles negative delays.
    neg_delays = cc[-max_tau:]
    pos_delays = cc[:max_tau + 1]
    cc_bounded = np.concatenate((neg_delays, pos_delays))

    # Return delay mapped back to [-max_tau, max_tau] interval range
    return np.argmax(cc_bounded) - max_tau


def get_direction(mic_data):
    """
    Evaluates bounded inter-channel delays to estimate quadrant direction orientation.
    """
    mic1, mic2, mic3, mic4 = mic_data[0], mic_data[1], mic_data[2], mic_data[3]

    # Query narrow search windows to eliminate cyclic wave frequency interference
    delay_y = gcc_phat(mic1, mic3, max_tau=10)
    delay_x = gcc_phat(mic2, mic4, max_tau=10)

    # Determine axis dominance and apply sign mapping vector profiles
    if abs(delay_y) > abs(delay_x):
        return ("⬆️ Front", 0.0) if delay_y < 0 else ("⬇️ Rear", math.pi)
    else:
        return ("⬅️ Left", -math.pi / 2) if delay_x < 0 else ("➡️ Right", math.pi / 2)


def calibrate_noise_floor(stream):
    print("\n[Environment Scan] Please remain quiet while collecting ambient noise...")
    frames = []

    start_time = time.time()

    for _ in range(5):
        stream.read(CHUNK, exception_on_overflow=False)

    while time.time() - start_time < 2.0:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        frames.append(
            librosa.feature.melspectrogram(
                y=audio_chunk[0::CHANNELS],
                sr=RATE,
                n_mels=128,
                n_fft=1024,
                hop_length=512
            )
        )

    print("[Scan Complete] Ambient noise profile captured.\n")

    return np.mean(
        np.concatenate(frames, axis=1),
        axis=1,
        keepdims=True
    )


def estimate_distance(audio_segment, noise_profile, sound_type):
    mel_spec = librosa.feature.melspectrogram(
        y=audio_segment,
        sr=RATE,
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )

    clean_mel_spec = np.maximum(mel_spec - noise_profile, 0.0)

    frame_power = np.mean(clean_mel_spec, axis=1)

    mel_freqs = librosa.mel_frequencies(
        n_mels=128,
        fmin=0.0,
        fmax=RATE / 2.0
    )

    e0 = np.sum(frame_power[(mel_freqs >= 100) & (mel_freqs < 500)])
    e3 = np.sum(frame_power[(mel_freqs >= 500) & (mel_freqs <= 1000)])
    e6 = np.sum(frame_power[(mel_freqs >= 2000) & (mel_freqs <= 3000)])

    if sound_type == "SIREN":
        ratio = (e3 / e6) if e6 >= 0.0001 else 9999.0

        if ratio < SIREN_NEAR_THRESH:
            return "Near", f"E3/E6: {ratio:.1f}"
        elif ratio > SIREN_FAR_THRESH:
            return "Far", f"E3/E6: {ratio:.1f}"

        return "Medium", f"E3/E6: {ratio:.1f}"

    else:
        ratio = (e0 / e3) if e3 >= 0.0001 else 9999.0

        if ratio < HORN_NEAR_THRESH:
            return "Near", f"E0/E3: {ratio:.4f}"
        elif ratio > HORN_FAR_THRESH:
            return "Far", f"E0/E3: {ratio:.4f}"

        return "Medium", f"E0/E3: {ratio:.4f}"


# Hardware and model initialization
p = pyaudio.PyAudio()

device_index = next(
    (
        i for i in range(p.get_device_count())
        if "SEEED" in p.get_device_info_by_index(i).get('name', '').upper()
           or "RESPEAKER" in p.get_device_info_by_index(i).get('name', '').upper()
    ),
    None
)

try:
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK
    )

except Exception as e:
    print(f"❌ Failed to open microphone: {e}")
    exit()

session = ort.InferenceSession("ressiren_final.onnx")
input_name = session.get_inputs()[0].name

# Global state
multi_buffer = np.zeros((4, BUFFER_SIZE), dtype=np.float32)
sample_counter = 0
noise_profile = None


# Core asynchronous processing loop
async def audio_processing_loop(websocket):
    global multi_buffer, sample_counter, noise_profile

    print("🟢 [Network] Quest 3 WebXR frontend connected. Starting real-time streaming and detection...")

    try:
        while True:

            # Capture real-time audio stream
            data = stream.read(CHUNK, exception_on_overflow=False)

            audio_chunk = np.frombuffer(
                data,
                dtype=np.int16
            ).astype(np.float32) / 32768.0

            # 4-channel unpacking and sliding window update
            mic1 = audio_chunk[0::CHANNELS]
            chunk_len = len(mic1)

            multi_buffer = np.roll(
                multi_buffer,
                -chunk_len,
                axis=1
            )

            multi_buffer[0, -chunk_len:] = mic1
            multi_buffer[1, -chunk_len:] = audio_chunk[1::CHANNELS]
            multi_buffer[2, -chunk_len:] = audio_chunk[2::CHANNELS]
            multi_buffer[3, -chunk_len:] = audio_chunk[3::CHANNELS]

            sample_counter += chunk_len

            # Run AI inference every 0.5 seconds
            if sample_counter >= PROCESS_INTERVAL:

                sample_counter = 0

                # Compute Mel spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=multi_buffer[0],
                    sr=RATE,
                    n_mels=128,
                    n_fft=1024,
                    hop_length=512
                )

                # --- Send spectrum waterfall data to Quest 3 ---
                current_power = np.mean(mel_spec, axis=1)
                fft_64 = current_power.reshape(64, 2).mean(axis=1)

                # Fixed absolute reference configuration scaling
                log_pow = librosa.power_to_db(
                    fft_64,
                    ref=1.0
                )

                MIN_DB = -60
                MAX_DB = 10

                fft_normalized = np.clip(
                    (log_pow - MIN_DB) / (MAX_DB - MIN_DB) * 255,
                    0,
                    255
                ).astype(int).tolist()

                await websocket.send(
                    json.dumps({
                        "msg_type": "FFT_STREAM",
                        "data": fft_normalized
                    })
                )

                # --- AI inference and alert generation ---
                log_mel_spec = librosa.power_to_db(
                    mel_spec,
                    ref=np.max
                )

                TARGET_FRAMES = 126

                if log_mel_spec.shape[1] > TARGET_FRAMES:
                    log_mel_spec = log_mel_spec[:, -TARGET_FRAMES:]

                elif log_mel_spec.shape[1] < TARGET_FRAMES:
                    pad_width = TARGET_FRAMES - log_mel_spec.shape[1]

                    log_mel_spec = np.pad(
                        log_mel_spec,
                        ((0, 0), (0, pad_width)),
                        mode='constant',
                        constant_values=-80.0
                    )

                predictions = session.run(
                    None,
                    {
                        input_name:
                            log_mel_spec[np.newaxis, np.newaxis, :, :].astype(np.float32)
                    }
                )[0][0]

                pred_id = np.argmax(predictions)
                confidence = np.max(predictions)

                if confidence > 0.85:

                    recent_audio = multi_buffer[:, -DISTANCE_WINDOW:]

                    # Direction estimation
                    dir_label, angle_rad = get_direction(recent_audio)

                    # Distance estimation and alert generation
                    alert_type = None

                    if pred_id == TARGET_HORN_ID:

                        dist_label, ratio_str = estimate_distance(
                            recent_audio[0],
                            noise_profile,
                            "HORN"
                        )

                        alert_type = "HONK"

                        print(
                            f"🚗 [HORN] Distance: {dist_label:<6} | "
                            f"Direction: {dir_label:<8} | "
                            f"AI: {confidence * 100:.1f}%"
                        )

                    elif pred_id == TARGET_SIREN_ID:

                        dist_label, ratio_str = estimate_distance(
                            recent_audio[0],
                            noise_profile,
                            "SIREN"
                        )

                        alert_type = "SIREN"

                        print(
                            f"🚨 [SIREN] Distance: {dist_label:<6} | "
                            f"Direction: {dir_label:<8} | "
                            f"AI: {confidence * 100:.1f}%"
                        )

                    if alert_type:
                        await websocket.send(
                            json.dumps({
                                "msg_type": "ALERT",
                                "type": alert_type,
                                "distance": dist_label,
                                "angle": angle_rad
                            })
                        )

                        # Clear buffer to avoid repeated alerts
                        multi_buffer.fill(0)

            # Yield to event loop to prevent WebSocket disconnects
            await asyncio.sleep(0.005)

    except websockets.exceptions.ConnectionClosed:
        print("🔴 [Network] WebXR frontend disconnected")

    except Exception as e:
        print(f"⚠️ [Error] Runtime exception: {e}")


async def main():
    global noise_profile

    # Collect ambient noise profile before startup
    noise_profile = calibrate_noise_floor(stream)

    print("=== Real-Time Microphone AI Edge Node Started (0.0.0.0:8765) ===")
    print("Please refresh the webpage inside Quest 3...")

    # Start WebSocket server
    async with websockets.serve(
            audio_processing_loop,
            "0.0.0.0",
            8765
    ):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\n⏹️ Test stopped")

    finally:
        if stream.is_active():
            stream.stop_stream()
            stream.close()

        p.terminate()