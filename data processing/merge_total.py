"""
Step 3: merge all CSVs into total_mel_index.csv for training
        synthesis_metadata_log.csv  (mixed samples)
      + processed_urbansound8k_single_npy.csv (urbansound single horn/siren)
      + processed_record.csv (self-recorded single horn/siren)
        → total_mel_index.csv
"""
import pandas as pd

SYNTHESIS_CSV = r"C:\Users\Petra\Desktop\ujm\projects\Code(cleaned\synthesis_metadata_log.csv"
SINGLE_CSV    = r"C:\Users\Petra\Desktop\ujm\projects\Code(cleaned\processed_urbansound8k_single_npy.csv"
RECORD_CSV    = r"C:\Users\Petra\Desktop\ujm\projects\Code(cleaned\processed_record.csv"
OUTPUT_CSV    = r"C:\Users\Petra\Desktop\ujm\projects\Code(cleaned\total_mel_index.csv"

# ── synthesis: already has correct columns ────────────────────────────────────
df_mix = pd.read_csv(SYNTHESIS_CSV)
df_mix = df_mix[['file_name', 'fold', 'final_label', 'clean_npy_name']].copy()

# ── urbansound single: only horn(classID=1) and siren(classID=8) ──────────────
df_single = pd.read_csv(SINGLE_CSV)
df_single = df_single[df_single['classID'].isin([1, 8])].copy()
df_single = df_single.rename(columns={'slice_file_name': 'file_name'})
df_single['file_name']      = df_single['file_name'].str.replace('.wav', '.npy', regex=False)
df_single['clean_npy_name'] = df_single['file_name']   # itself is clean GT
df_single = df_single[['file_name', 'fold', 'final_label', 'clean_npy_name']].copy()

# ── record: all are horn/siren ────────────────────────────────────────────────
df_record = pd.read_csv(RECORD_CSV)
df_record = df_record.rename(columns={'file_name': 'file_name'})
df_record = df_record[['file_name', 'fold', 'final_label', 'clean_npy_name']].copy()

# ── merge ─────────────────────────────────────────────────────────────────────
df_total = pd.concat([df_mix, df_single, df_record], ignore_index=True)
df_total.to_csv(OUTPUT_CSV, index=False)

print(f"Total rows: {len(df_total)}")
print(df_total['final_label'].value_counts())
print(f"\nSaved: {OUTPUT_CSV}")