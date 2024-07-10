import pandas as pd

csv_file_path = 'ptbxl_database.csv'
df = pd.read_csv(csv_file_path)
# print(df)

# Example of data extraction based on ecg_id
ecg_id = 1
recording_date = df.loc[df['ecg_id'] == ecg_id, 'recording_date'].values[0]
print(f"ecg_id = {ecg_id} has associated recording date: {recording_date}")