from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from audiomentations import AddBackgroundNoise, Normalize, Compose

total_mel = Path(r"D:\Petra\UrbanSound8K\UrbanSound8K\total_mel_dataset")
mix_wav   = Path(r"D:\Petra\UrbanSound8K\UrbanSound8K\mixoutput_wave")

for folder in [total_mel, mix_wav]:
    count = 0
    for fold_dir in folder.glob("fold*"):
        for f in fold_dir.glob("mix_*.npy"):
            f.unlink()
            count += 1
        for f in fold_dir.glob("mix_*.wav"):
            f.unlink()
            count += 1
    print(f"Deleted {count} files from {folder}")

RECORD_FOLDS = {1, 3, 5, 7, 9}
RECORD_COUNT = 500


class AudioMixer:
    def __init__(self, csv_path, audio_root, noise_root,
                 output_npy_dir, wav_output_dir, sr=16000, duration=4.0):
        self.df             = pd.read_csv(csv_path)
        self.audio_root     = Path(audio_root)
        self.noise_root     = noise_root
        self.output_npy_dir = Path(output_npy_dir)
        self.wav_output_dir = Path(wav_output_dir)
        self.sr             = sr
        self.duration       = duration
        self.target_samples = int(self.sr * self.duration)

        self.output_npy_dir.mkdir(parents=True, exist_ok=True)
        self.wav_output_dir.mkdir(parents=True, exist_ok=True)

    def get_label(self, class_id_list):
        if 1 in class_id_list:
            return 1
        elif 8 in class_id_list:
            return 2
        else:
            return 3

    def load_audio(self, row):
        source = str(row.get('source', 'urbansound'))

        if source == 'record':
            # directly load saved wav — fast, no Griffin-Lim needed
            wav_path = str(row.get('wav_path', ''))
            if not wav_path or not Path(wav_path).exists():
                print(f"  record wav not found: {wav_path}")
                return np.zeros(self.target_samples, dtype=np.float32)
            try:
                audio, _ = librosa.load(wav_path, sr=self.sr)
                if len(audio) < self.target_samples:
                    audio = np.pad(audio, (0, self.target_samples - len(audio)), mode='constant')
                else:
                    audio = audio[:self.target_samples]
                return audio.astype(np.float32)
            except Exception as e:
                print(f"Error loading record wav {wav_path}: {e}")
                return np.zeros(self.target_samples, dtype=np.float32)
        else:
            fold_name = f"fold{int(row['fold'])}"
            filename  = str(row['slice_file_name'])
            if filename.endswith('.npy'):
                filename = filename[:-4] + '.wav'
            file_path = self.audio_root / fold_name / filename
            try:
                audio, _ = librosa.load(str(file_path), sr=self.sr)
                if len(audio) < self.target_samples:
                    audio = np.pad(audio, (0, self.target_samples - len(audio)), mode='constant')
                else:
                    audio = audio[:self.target_samples]
                return audio.astype(np.float32)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                return np.zeros(self.target_samples, dtype=np.float32)

    def get_augmentation_pipeline(self):
        return Compose([
            AddBackgroundNoise(
                sounds_path=self.noise_root,
                min_snr_db=10.0,
                max_snr_db=25.0,
                p=0.6
            ),
            Normalize(p=1.0)
        ])

    def _mix_one(self, anchor_pool, other_pool, target_label, mix_options, mix_probs):
        num_extra = np.random.choice(mix_options, p=mix_probs) - 1

        if target_label in (1, 2):
            anchor     = anchor_pool.sample(n=1, replace=True)
            background = other_pool.sample(n=num_extra, replace=True) \
                         if len(other_pool) > 0 and num_extra > 0 else pd.DataFrame()
            batch      = pd.concat([anchor, background], ignore_index=True)
        else:
            n_total = num_extra + 1
            batch   = other_pool.sample(n=n_total, replace=True) \
                      if len(other_pool) > 0 else anchor_pool.sample(n=1, replace=True)

        mixed_audio    = np.zeros(self.target_samples, dtype=np.float32)
        class_ids      = []
        horn_npy_name  = ""
        siren_npy_name = ""

        for _, row in batch.iterrows():
            audio_segment = self.load_audio(row)
            gain          = np.random.uniform(0.5, 0.9)
            mixed_audio  += audio_segment * gain
            cid           = int(row['classID'])
            class_ids.append(cid)

            if cid == 1 and horn_npy_name == "":
                horn_npy_name = Path(row['npy_path']).name \
                                if pd.notna(row.get('npy_path', '')) else \
                                f"{Path(row['slice_file_name']).stem}.npy"
            if cid == 8 and siren_npy_name == "":
                siren_npy_name = Path(row['npy_path']).name \
                                 if pd.notna(row.get('npy_path', '')) else \
                                 f"{Path(row['slice_file_name']).stem}.npy"

        return mixed_audio, class_ids, horn_npy_name, siren_npy_name

    def synthesis(self, total_count_per_fold=1500, label_ratio=(3, 2, 1)):
        results     = []
        mix_options = [2, 3, 4, 5]
        mix_probs   = [0.4, 0.3, 0.2, 0.1]
        pipeline    = self.get_augmentation_pipeline()

        ratio_sum   = sum(label_ratio)
        count_horn  = int(total_count_per_fold * label_ratio[0] / ratio_sum)
        count_siren = int(total_count_per_fold * label_ratio[1] / ratio_sum)
        count_other = total_count_per_fold - count_horn - count_siren

        print(f"Per fold: horn={count_horn}  siren={count_siren}  "
              f"other={count_other}  total={total_count_per_fold}")

        all_folds = sorted(self.df['fold'].unique())

        for fold_id in all_folds:
            print(f"\nProcessing Fold {fold_id}...")
            fold_df = self.df[self.df['fold'] == fold_id]

            # pools
            record_horn  = fold_df[(fold_df['source'] == 'record') & (fold_df['classID'] == 1)]
            record_siren = fold_df[(fold_df['source'] == 'record') & (fold_df['classID'] == 8)]
            urban_horn   = fold_df[(fold_df['source'] != 'record') & (fold_df['classID'] == 1)]
            urban_siren  = fold_df[(fold_df['source'] != 'record') & (fold_df['classID'] == 8)]
            other_pool   = fold_df[~fold_df['classID'].isin([1, 8])]

            fold_npy_path = self.output_npy_dir / f"fold{fold_id}"
            fold_wav_path = self.wav_output_dir / f"fold{fold_id}"
            fold_npy_path.mkdir(parents=True, exist_ok=True)
            fold_wav_path.mkdir(parents=True, exist_ok=True)

            # ── build schedule ────────────────────────────────────────────────
            if fold_id in RECORD_FOLDS and len(record_horn) > 0:
                rec_horn_count  = int(RECORD_COUNT * label_ratio[0] / (label_ratio[0] + label_ratio[1]))
                rec_siren_count = RECORD_COUNT - rec_horn_count
                record_schedule = [1] * rec_horn_count + [2] * rec_siren_count
                np.random.shuffle(record_schedule)

                normal_total    = total_count_per_fold - RECORD_COUNT
                n_horn  = int(normal_total * label_ratio[0] / ratio_sum)
                n_siren = int(normal_total * label_ratio[1] / ratio_sum)
                n_other = normal_total - n_horn - n_siren
                normal_schedule = [1] * n_horn + [2] * n_siren + [3] * n_other
                np.random.shuffle(normal_schedule)

                print(f"  record section : {len(record_schedule)} "
                      f"(horn={rec_horn_count} siren={rec_siren_count})")
                print(f"  normal section : {len(normal_schedule)} "
                      f"(horn={n_horn} siren={n_siren} other={n_other})")
            else:
                record_schedule = []
                n_horn  = count_horn
                n_siren = count_siren
                n_other = count_other
                normal_schedule = [1] * n_horn + [2] * n_siren + [3] * n_other
                np.random.shuffle(normal_schedule)
            # ─────────────────────────────────────────────────────────────────

            all_schedule = record_schedule + normal_schedule

            for i, target_label in enumerate(tqdm(all_schedule, desc=f"Fold {fold_id}")):
                use_record = (i < len(record_schedule))

                if use_record:
                    horn_anchor  = record_horn  if len(record_horn)  > 0 else urban_horn
                    siren_anchor = record_siren if len(record_siren) > 0 else urban_siren
                else:
                    horn_anchor  = pd.concat([urban_horn,  record_horn],  ignore_index=True) \
                                   if len(record_horn) > 0 else urban_horn
                    siren_anchor = pd.concat([urban_siren, record_siren], ignore_index=True) \
                                   if len(record_siren) > 0 else urban_siren

                if target_label == 1:
                    anchor_pool = horn_anchor
                elif target_label == 2:
                    anchor_pool = siren_anchor
                else:
                    anchor_pool = other_pool

                mixed_audio, class_ids, horn_npy_name, siren_npy_name = self._mix_one(
                    anchor_pool, other_pool, target_label, mix_options, mix_probs
                )

                final_audio = pipeline(samples=mixed_audio, sample_rate=self.sr)
                final_label = self.get_label(class_ids)
                filename    = f"mix_{i:04d}_{fold_id}_{final_label}"

                sf.write(str(fold_wav_path / f"{filename}.wav"), final_audio, self.sr)

                mel_spec     = librosa.feature.melspectrogram(
                    y=final_audio, sr=self.sr, n_fft=1024, hop_length=512, n_mels=128
                )
                log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                np.save(str(fold_npy_path / f"{filename}.npy"), log_mel_spec)

                clean_npy_name = horn_npy_name  if final_label == 1 else \
                                 siren_npy_name if final_label == 2 else ""

                results.append({
                    "file_name"     : f"{filename}.npy",
                    "fold"          : fold_id,
                    "final_label"   : final_label,
                    "mixed_classes" : class_ids,
                    "clean_npy_name": clean_npy_name,
                    "use_record"    : use_record,
                })

        log_df = pd.DataFrame(results)
        log_df.to_csv(
            r"C:\Users\Petra\Desktop\ujm\projects\Code(cleaned\synthesis_metadata_log.csv",
            index=False
        )
        print("\nSynthesis completed.")
        print(log_df['final_label'].value_counts())
        print(f"Record-anchored samples: {log_df['use_record'].sum()}")


if __name__ == "__main__":
    mixer = AudioMixer(
        csv_path        = r"C:\Users\Petra\Desktop\ujm\projects\Code(cleaned\combined_source.csv",
        audio_root      = r"D:\Petra\UrbanSound8K\UrbanSound8K\audio",
        noise_root      = r"D:\Petra\UrbanSound8K\UrbanSound8K\noise",
        output_npy_dir  = r"D:\Petra\UrbanSound8K\UrbanSound8K\total_mel_dataset",
        wav_output_dir  = r"D:\Petra\UrbanSound8K\UrbanSound8K\mixoutput_wave",
    )
    mixer.synthesis(total_count_per_fold=1500, label_ratio=(3, 2, 1))