import wfdb
from pathlib import Path
import csv
import pandas as pd

# Define the directory containing the patient folders
root_dir = Path(__file__).resolve().parent.parent
dataPath = root_dir / "physionet.org/files"

# Use rglob to find the main CSV file
csv_files = list(dataPath.rglob("ptbxl_database.csv"))[0]

# Define the output data structure
data_records = []

# Define the path to the raw ECG signal records
ecg_data_path = root_dir / "physionet.org/files/ptb-xl/1.0.3"

# Iterate through the rows in the CSV database
with open(csv_files) as file_obj:
    reader_obj = csv.reader(file_obj)
    headers = next(reader_obj)  # Read the header row
    filename_lr_index = headers.index("filename_lr")  # Get the index of filename_lr

    for row in reader_obj:
        patient_id = row[1]
        report = row[10]
        scp_codes = row[11]
        filename_lr = row[filename_lr_index]  # Extract the filename_lr

        # Define the path to the corresponding .hea file
        hea_file_path = ecg_data_path / filename_lr
        if not hea_file_path.with_suffix(".hea").exists():
            print(f"Error: .hea file {hea_file_path} does not exist.")
            continue

        try:
            # Extract the study number from the file name
            study_num = hea_file_path.stem
            print(f"Processing file {hea_file_path}")

            # Read the raw ECG signal data
            # .hea files contain header information while .dat files and others contain the signal data
            record = wfdb.rdrecord(str(hea_file_path.with_suffix("")))
            ecg_data = record.p_signal

            # Print out the attributes of the record object [useful if you want to see whats inside the wfdb record]
            # attributes = [
            #     attr
            #     for attr in dir(record)
            #     if not callable(getattr(record, attr)) and not attr.startswith("__")
            # ]
            # for attr in attributes:
            #     print(f"{attr}: {getattr(record, attr)}")

            # Append the data to the output structure
            data_records.append(
                {
                    "study_num": study_num,
                    "patient_id": patient_id,
                    "report": report,
                    "scp_codes": scp_codes,
                    "ecg_data": ecg_data.tolist(),  # Convert numpy array to list for serialization
                }
            )
            break
        except Exception as e:
            print(f"Error processing file {hea_file_path}: {e}")
            continue

# Save the collected data into a structured format
output_file = root_dir / "ecg_data.json"
pd.DataFrame(data_records).to_json(output_file, orient="records")

print(f"Data has been saved to {output_file}")

import pprint

with open(output_file, "r") as file_content:
    file_content = file_content.readlines()
    pprint.pp(file_content)
