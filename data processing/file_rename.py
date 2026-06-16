import pandas as pd

df = pd.read_csv(r"C:\Users\Petra\Desktop\ujm\projects\Code(cleaned\processed_urbansound8k_single.csv")

df['slice_file_name'] = df['slice_file_name'].str.replace('.wav', '.npy', regex=False)

df.to_csv(r"C:\Users\Petra\Desktop\ujm\projects\Code(cleaned\processed_urbansound8k_single_npy.csv", index=False)

print("Done.")

