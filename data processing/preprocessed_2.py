import librosa
import librosa.display
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def preprocess_audio(file_path, target_sr=16000, n_mels=128):
    try:
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=sr // 2)
        log_S = librosa.power_to_db(S, ref=np.max)
        return log_S
    except Exception as e:
        print(f"Error processing {file_path}:{e}")
        return None

def visualization_mel_spectrogram(log_S, target_sr=16000):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_S, sr=target_sr, x_axis='time', y_axis='mel', fmax=target_sr // 2)
    plt.colorbar(format="%+2.0f dB")
    plt.show()

print("Now process selected files!")

AUDIO_ROOT   = Path("D:/Petra/UrbanSound8K/UrbanSound8K/audio")
METADATA_CSV = Path("D:/Petra/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv")
SAVE_DIR     = Path("D:/Petra/UrbanSound8K/UrbanSound8K/total_mel_dataset")
TARGET_DIR   = Path("D:/Petra/UrbanSound8K/UrbanSound8K/target_mel_dataset")

SAVE_DIR.mkdir(parents=True, exist_ok=True)
TARGET_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_CLASSES = [1, 3, 4, 5, 8, 9]
TARGET_CLASSES   = {1, 8}

CLASS_TO_LABEL = {1: 1, 8: 2}  # ← 新增：classID → final_label 映射

df = pd.read_csv(METADATA_CSV)
df_selected = df[df['classID'].isin(SELECTED_CLASSES)].copy()
print(f"Totally, {len(df_selected)} files need to be process!")

processed_info = []

for index, row in tqdm(df_selected.iterrows(), total=df_selected.shape[0]):
    fold_name = f"fold{row['fold']}"
    file_path = AUDIO_ROOT / fold_name / row['slice_file_name']

    if not file_path.exists():
        print(f"File is not exist: {file_path}")
        continue

    mel_spec = preprocess_audio(str(file_path))
    if mel_spec is None:
        continue

    fold_save = SAVE_DIR / fold_name
    fold_save.mkdir(parents=True, exist_ok=True)

    npy_filename = f"{Path(file_path).stem}.npy"
    npy_path     = fold_save / npy_filename
    np.save(npy_path, mel_spec)

    row_dict = row.to_dict()
    row_dict['npy_path']    = str(npy_path)
    row_dict['final_label'] = CLASS_TO_LABEL.get(int(row['classID']), 3)  # ← 新增

    processed_info.append(row_dict)

    if int(row['classID']) in TARGET_CLASSES:
        fold_target = TARGET_DIR / fold_name
        fold_target.mkdir(parents=True, exist_ok=True)
        np.save(fold_target / npy_filename, mel_spec)

df_final = pd.DataFrame(processed_info)
df_final.to_csv("processed_urbansound8k_single.csv", index=False)
print("Process Successfully!")