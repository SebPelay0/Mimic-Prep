import os
import sys
import re
import gc
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import wfdb

# import psutil
import pandas as pd


"""This class initialises sample data folders and reads through the PTB-XL database
currently, it is able generate images for the 10 most commonnly occuring SCP codes in the database. 
The number of samples and arrythmias can be changed as needed. """


class DataLoader:
    def __init__(self):
        # Initialize file paths and parameters

        self.root_dir = Path(__file__).resolve().parent.parent
        self.dataPath = self.root_dir / "physionet.org/files"
        self.imagesPath = self.root_dir / "data"
        self.ecg_data_path = self.root_dir / "physionet.org/files/mimic3/p01"
        # self.csv_file = self.openPTBDataset()
        self.classes = [
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
        self.chunksize = 1000
        # where chunksize is how much of the csv we read at a time, adjust for perfomance issues.
        self.maxSamples = 100
        self.printMembers()
        # show startup conditions

    def printMembers(self):
        print("DataLoader Initialized with the following attributes:")
        for attribute, value in self.__dict__.items():
            print(f"{attribute}: {value}")

    # newClass should be a string in the form "CRBB" idk how to show that in python
    def addArrythmiaToClasses(self, newClass):
        self.classes.append(newClass)

    def SetupClassFolders(self):
        """ONLY USE IF FOLDERS HAVE NOT ALREADY BEEN SET"""
        for label in self.classes:
            class_path = self.imagesPath / label.strip()
            class_path.mkdir(parents=True, exist_ok=True)

    def openPTBDataset(self):
        # Locate the main CSV file, should be installed in this path upon git pull
        csv_files = list(self.dataPath.rglob("ptbxl_database.csv"))
        if not csv_files:
            print("Error: CSV file not found")
            sys.exit(1)
        return csv_files[0]

    def setSampleSize(self, maxSamples):
        """Sets a max limit on the number of samples in each folder"""
        self.maxSamples = maxSamples

    def checkIfAtSampleLimit(self):
        for label in self.classes:
            class_path = self.imagesPath / label.strip()
            if len(list(class_path.glob("*.png"))) < self.maxSamples:
                return False
            return True

    # def checkMemUsage():
    #     """Get current memory usage. Used to sanity check if loading process has stopped"""
    #     process = psutil.Process(os.getpid())
    #     return process.memory_info().rss / 1024 / 1024  # Memory in MB

    def LoadData(self):
        """Read through the CSV, filter, and process each row to generate images for selected SCP codes."""
        for chunk in pd.read_csv(self.csv_file, chunksize=self.chunksize):
            for index, row in chunk.iterrows():
                # Extract and parse SCP code
                scp_code = row.iloc[11]
                filename_lr = row["filename_lr"]

                # Define the path to the corresponding .hea file
                hea_file_path = self.ecg_data_path / filename_lr
                if not hea_file_path.with_suffix(".hea").exists():
                    print(f"Error: .hea file {hea_file_path} does not exist.")
                    continue

                try:
                    # Extract the SCP code with the highest confidence value
                    scp_code_matches = re.findall(r"(\w+)':\s*(\d+\.\d+)", scp_code)
                    useful_scp_code = max(scp_code_matches, key=lambda x: float(x[1]))
                    highest_scp_code, value = useful_scp_code

                    """This section here can be configured to tell the obj to only to fill up folders of one SCP type"""
                    # Skip if the class is not in the desired list
                    # if 'ASMI' not in highest_scp_code:
                    #     continue

                    output_dir = self.imagesPath / highest_scp_code.strip()

                    # Check if the directory is at sample limit yet
                    if len(list(output_dir.glob("*.png"))) >= self.maxSamples:
                        print(
                            f"Directory {output_dir} already has 100 images. Skipping."
                        )
                        continue

                    # Extract the study number from the file name
                    study_num = hea_file_path.stem
                    print(f"Processing file {hea_file_path}")

                    # Read the raw ECG signal data
                    rd_record = wfdb.rdrecord(str(hea_file_path.with_suffix("")))
                    ecg_data = rd_record.p_signal

                    # Plot and save the ECG data
                    fig = wfdb.plot_wfdb(
                        record=rd_record,
                        figsize=(24, 18),
                        title=row.iloc[1],
                        ecg_grids="all",
                        return_fig=True,
                    )

                    output_file_path = output_dir / f"{study_num}.png"
                    fig.savefig(output_file_path)
                    plt.close(fig)
                    gc.collect()  # Trigger garbage collection

                    print(f"Saved image to {output_file_path}")
                except Exception as e:
                    print(f"Error processing file {hea_file_path}: {e}")
                    continue

                ##TODO: Change this so it doesnt run on every loop, prob slows it down a lot
                if self.checkIfAtSampleLimit():
                    print("All directories have at least 100 images. Stopping.")
                    return

    def loadMimic3Data(self):
        """Retrieves the MIMIC-III specific .dat and .hea files and saves ECG/ABP/PAP plots."""
        counter = 0
        input_directory = Path("physionet.org/files/mimic3/p01/p011279")
        output_directory = Path("/Users/sky/Desktop/COMP/VIP/Mimic-Prep/data")

        for file in os.listdir(input_directory):
            try:
                if ".dat" not in file and ".hea" not in file:
                    continue

                file_stem = file.split(".")[0]
                file_path = input_directory / file_stem

                # Read ECG/ABP/PAP data
                rd_record = wfdb.rdrecord(str(file_path))

                # Display available signals and their names
                print(f"Signals: {rd_record.sig_name}")
                print(f"Shape of data: {rd_record.p_signal.shape}")

                # Plot and save the signals (adjusting the grid and style for each channel)
                fig = wfdb.plot_wfdb(
                    record=rd_record,
                    figsize=(24, 18),
                    title="file",
                    ecg_grids=None,
                    return_fig=True,
                )

                output_file_path = output_directory / f"{counter}.png"
                fig.savefig(output_file_path)
                plt.close(fig)
                gc.collect()  # Trigger garbage collection if needed

                print(f"Saved image to {output_file_path}")
                counter += 1

            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue


testDataLoader = DataLoader()
# testDataLoader.addArrythmiaToClasses("TEST");
# testDataLoader.printMembers()
# testDataLoader.LoadData()
testDataLoader.loadMimic3Data()
