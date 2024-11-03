import os
import gc
from pathlib import Path
import wfdb
import matplotlib.pyplot as plt

"""
Python data loader that turns files downloaded from https://physionet.org/content/mimic-iv-ecg-demo/0.1/
and converts them into png images.

If a different database is used in future replace self.mimicDataPath with the new path

Other databases that can be used https://physionet.org/content/mimic-iv-ecg/1.0/files/
"""


class DataLoader:
    def __init__(self):
        # Initialize file paths and parameters
        self.root_dir = Path(__file__).resolve().parent.parent
        self.mimicDataPath = "mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1/files"
        self.imagesPath = self.root_dir / "data"

    def get_header_files(self, input_directory):
        """Generator to yield file paths for .hea files."""
        for dir in os.listdir(input_directory):
            for subdir in os.listdir(os.path.join(input_directory, dir)):
                path = os.path.join(input_directory, dir, subdir)
                for file in os.listdir(path):
                    if file.endswith(".hea"):
                        yield path, file

    def plot_and_save_record(self, file_path, output_directory, counter):
        """Reads the record, plots it, and saves the plot to an image file."""
        try:
            title = str(file_path).split("/")[-1]
            rd_record = wfdb.rdrecord(str(file_path))

            # Display signal details for debugging
            print(f"Signals: {rd_record.sig_name}")
            print(f"Shape of data: {rd_record.p_signal.shape}")

            # Plot and save the figure
            fig = wfdb.plot_wfdb(
                record=rd_record,
                figsize=(24, 18),
                title=f"Record {title}",
                ecg_grids=None,
                return_fig=True,
            )

            output_directory.mkdir(parents=True, exist_ok=True)
            output_file_path = output_directory / f"{counter}.png"
            fig.savefig(output_file_path)
            plt.close(fig)
            gc.collect()  # Trigger garbage collection if needed

            print(f"Saved image to {output_file_path}")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    def loadMimic4Data(self):
        """Retrieves the MIMIC-IV specific data files and saves ECG  plots."""
        counter = 0
        input_directory = self.mimicDataPath
        output_directory = self.imagesPath

        for path, file in self.get_header_files(input_directory):
            file_stem = file.split(".")[0]
            file_path = os.path.join(path, file_stem)
            output_directory_path = output_directory / file_stem

            # Plot and save the ECG data
            self.plot_and_save_record(file_path, output_directory_path, counter)
            counter += 1


if __name__ == "__main__":
    data_loader = DataLoader()
    data_loader.loadMimic4Data()
