#!/usr/bin/env python3

import os
import sys
import re
import gc
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import wfdb
import psutil
import pandas as pd
# Define the directory containing the patient folders
root_dir = Path(__file__).resolve().parent.parent
dataPath = root_dir / "physionet.org/files"
imagesPath = root_dir / "data"

# Define the SCP codes
classes = [
    "NORM",
    "IMI",
    "NDT",
    "ASMI",
    "LVH",
    "LAFB",
    "IRBBB",
    "CLBBB",
    "NST_",
    "CRBBB",
]

# Create folders for each SCP code if they don't exist
for label in classes:
    class_path = imagesPath / label.strip()
    class_path.mkdir(parents=True, exist_ok=True)


# Use rglob to find the main CSV file
csv_files = list(dataPath.rglob("ptbxl_database.csv"))
if not csv_files:
    print("Error: CSV file not found")
    sys.exit(1)
csv_file = csv_files[0]

# Define the path to the raw ECG signal records
ecg_data_path = root_dir / "physionet.org/files/ptb-xl/1.0.3"


def all_directories_have_100_images(images_path, classes):
    """Check if all directories have at least 100 images."""
    for label in classes:
        class_path = images_path / label.strip()
        if len(list(class_path.glob("*.png"))) < 100:
            return False
    return True


def memory_usage():
    """Get current memory usage."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Memory in MB


# Iterate through the rows in the CSV database
chunksize = 1000  
for chunk in pd.read_csv(csv_file, chunksize=chunksize):
# with open(csv_file) as file_obj:
    for index, row in chunk.iterrows():
        # save only SCP
        
        scp_code = row[11]
        filename_lr = row['filename_lr']  # Extract the filename_lr

        # Define the path to the corresponding .hea file
        hea_file_path = ecg_data_path / filename_lr
        if not hea_file_path.with_suffix(".hea").exists():
            print(f"Error: .hea file {hea_file_path} does not exist.")
            continue
        
        try:
            # Define the output path for the image
            scp_code = re.findall(r"(\w+)':\s*(\d+\.\d+)", scp_code)
            useful_scp_code = max(scp_code, key=lambda x: float(x[1]))
            highest_scp_code, value = useful_scp_code
            if 'ASMI' not in highest_scp_code:
                continue
            output_dir = imagesPath / highest_scp_code.strip()

            # Check if the directory has 100 images already
            if len(list(output_dir.glob("*.png"))) >= 100:
                print(f"Directory {output_dir} already has 100 images. Skipping.")
                continue
            
            # Extract the study number from the file name
            study_num = hea_file_path.stem
            print(f"Processing file {hea_file_path}")
            print(f'Length of directory: {len(list(output_dir.glob("*.png")))}')
            # Read the raw ECG signal data
            rd_record = wfdb.rdrecord(str(hea_file_path.with_suffix("")))
            ecg_data = rd_record.p_signal

            # Plot and save the ECG data
            fig = wfdb.plot_wfdb(
                record=rd_record,
                figsize=(24, 18),
                title=row[1],
                ecg_grids="all",
                return_fig=True,
            )

            output_file_path = output_dir / f"{study_num}.png"
            fig.savefig(output_file_path)
            plt.close(fig)
            gc.collect()  # Explicitly trigger garbage collection

            print(f"Saved image to {output_file_path}")
        except Exception as e:
            print(f"Error processing file {hea_file_path}: {e}")
            continue

        # Check if all directories have 20 images
        if all_directories_have_100_images(imagesPath, classes):
            print("All directories have at least 100 images. Stopping.")
            break

        # Check memory usage
        # if memory_usage() > 1000:  # Example threshold in MB
        #     print("Memory usage is too high, stopping.")
        #     break
