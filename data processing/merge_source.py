"""
Step 1: merge processed_urbansound8k_single_npy.csv + processed_record.csv
        → combined_source.csv  (used by synthesis as audio pool)
"""
import pandas as pd
from pathlib import Path

SINGLE_CSV = r"C:\Users\Petra\Desktop\ujm\projects\Code(cleaned\processed_urbansound8k_single_npy.csv"
RECORD_CSV = r"C:\Users\Petra\Desktop\ujm\projects\Code(cleaned\processed_record.csv"
OUTPUT_CSV = r"C:\Users\Petra\Desktop\ujm\projects\Code(cleaned\combined_source.csv"

# ── Load single (UrbanSound8K) ────────────────────────────────────────────────
df_single = pd.read_csv(SINGLE_CSV)

df_single = df_single[['slice_file_name', 'fold', 'classID', 'final_label', 'npy_path']].copy()
df_single['source']         = 'urbansound'
df_single['wav_path']       = ''   # urbansound loads from audio_root, not wav_path
df_single['clean_npy_name'] = df_single.apply(
    lambda r: Path(r['npy_path']).name if r['final_label'] in (1, 2) else '', axis=1
)

# ── Load record (self-recorded) ───────────────────────────────────────────────
df_record = pd.read_csv(RECORD_CSV)

LABEL_TO_CLASSID = {1: 1, 2: 8}
df_record['classID']         = df_record['final_label'].map(LABEL_TO_CLASSID)
df_record['slice_file_name'] = df_record['file_name']
df_record['source']          = 'record'

df_record = df_record[['slice_file_name', 'fold', 'classID', 'final_label',
                        'npy_path', 'wav_path', 'clean_npy_name', 'source']].copy()

# ── Concatenate ───────────────────────────────────────────────────────────────
df_combined = pd.concat([df_single, df_record], ignore_index=True)
df_combined.to_csv(OUTPUT_CSV, index=False)

print(f"Combined rows : {len(df_combined)}")
print(df_combined['final_label'].value_counts())
print(df_combined['source'].value_counts())
print(f"Saved: {OUTPUT_CSV}")