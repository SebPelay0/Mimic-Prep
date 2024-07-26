import wfdb
from pathlib import Path
import csv
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

classes = [
    "NORM",
    "IMI",
    "NDT",
    "ASMI",
    " LVH",
    "LAFB",
    "IRBB",
    "CLBBB",
    "NST_",
    "CRBBB",
]
# Define the directory containing the patient folders
root_dir = Path(__file__).resolve().parent.parent
dataPath = root_dir / "physionet.org/files"

imagesPath = root_dir / "data"

# make folder for each of the 10 SCP codes
for label in classes:
    class_path = os.path.join(imagesPath, label.strip())
    if not (os.path.exists(class_path)):
        os.makedirs(class_path)
        print("made directory" + class_path)

# Use rglob to find the main CSV file
csv_files = list(dataPath.rglob("ptbxl_database.csv"))[0]

# Define the output data structure
data_records = []

# Define the path to the raw ECG signal records
ecg_data_path = root_dir / "physionet.org/files/ptb-xl/1.0.3"


def all_directories_have_20_images(images_path, classes):
    """Check if all directories have at least 20 images."""
    for label in classes:
        class_path = images_path / label.strip()
        if len(list(class_path.glob("*.png"))) < 20:
            return False
    return True


# Iterate through the rows in the CSV database
with open(csv_files) as file_obj:
    reader_obj = csv.reader(file_obj)
    headers = next(reader_obj)  # Read the header row
    filename_lr_index = headers.index("filename_lr")  # Get the index of filename_lr

    for row in reader_obj:
        # save only SCP
        scp_code = row[11]
        filename_lr = row[filename_lr_index]  # Extract the filename_lr

        # Define the path to the corresponding .hea file
        hea_file_path = ecg_data_path / filename_lr
        if not hea_file_path.with_suffix(".hea").exists():
            print(f"Error: .hea file {hea_file_path} does not exist.")
            continue

        try:
            # Check if the directory has 20 images already
            if len(list(output_dir.glob("*.png"))) >= 20:
                print(f"Directory {output_dir} already has 20 images. Skipping.")
                plt.close(fig)
                continue

            # Extract the study number from the file name
            study_num = hea_file_path.stem
            print(f"Processing file {hea_file_path}")

            # Read the raw ECG signal data
            # .hea files contain header information while .dat files and others contain the signal data
            rd_record = wfdb.rdrecord(str(hea_file_path.with_suffix("")))
            ecg_data = rd_record.p_signal

            # Print out the attributes of the record object [useful if you want to see whats inside the wfdb record]
            # attributes = [
            #     attr
            #     for attr in dir(record)
            #     if not callable(getattr(record, attr)) and not attr.startswith("__")
            # ]
            # for attr in attributes:
            #     print(f"{attr}: {getattr(record, attr)}")

            # row[1] indicates patientId
            # Append the data to the output structure
            fig = wfdb.plot_wfdb(
                record=rd_record,
                figsize=(24, 18),
                title=row[1],
                ecg_grids="all",
                return_fig=True,
            )

            # Define the output path for the image
            scp_code = re.findall(r"(\w+)':\s*(\d+\.\d+)", scp_code)
            useful_scp_code = max(scp_code, key=lambda x: float(x[1]))
            highest_scp_code, value = useful_scp_code
            output_dir = imagesPath / highest_scp_code.strip()

            output_file_path = output_dir / f"{study_num}.png"
            fig.savefig(output_file_path)
            plt.close(fig)

            print(f"Saved image to {output_file_path}")
        except Exception as e:
            print(f"Error processing file {hea_file_path}: {e}")
            continue

        # Check if all directories have 20 images
        if all_directories_have_20_images(imagesPath, classes):
            print("All directories have at least 20 images. Stopping.")
            break
