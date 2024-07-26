import wfdb
from pathlib import Path
import pandas as pd
import datetime
import os



classes = ['NORM', 'IMI', 'NDT', 'ASMI', ' LVH', 'LAFB', 'IRBB', 'CLBBB', 'NST_:', 'CRBBB']
# Define the directory containing the patient folders
root_dir = Path(__file__).resolve().parent.parent
dataPath = root_dir / "physionet.org/files"

imagesPath = root_dir/'data'

for label in classes:
    class_path = os.path.join(imagesPath, label.strip())
    if not (os.path.exists(class_path)):
        os.makedirs(class_path)
        print('made directory' + class_path)
        


print("Script started")

# Define the directory containing the patient folders
root_dir = Path(__file__).resolve().parent.parent
dataPath = root_dir / "SampleData"

# Use rglob to find all .hea files in the directory
hea_files = list(dataPath.rglob("*.hea"))

print(f"Found {len(hea_files)} .hea files")

studyData = {'studyNum':[], 'date':[], 'time':[]}
studyDates = []
studyTimes = []

for file in hea_files:
    print(f"Processing file {file}")

    filename = Path(file)
    filename = filename.with_suffix('')

    # Get file path to produce image
    filePath = file.with_suffix('')
    dat_file_path = filePath.with_suffix('.dat')

    # Check if the .dat file exists
    if not dat_file_path.exists():
        print(f"Error: .dat file {dat_file_path} does not exist.")
        continue

    try:
        rd_record = wfdb.rdrecord(str(filePath))
    except ValueError as e:
        print(f"Error reading record for {filePath}: {e}")
        continue

    fig = wfdb.plot_wfdb(record=rd_record, figsize=(24, 18), title=studyNum, ecg_grids='all', return_fig=True)
    image = fig.savefig(studyNum + '.png')
    filepath = os.path.join(imagesPath, scp_code)
    with open(filepath, 'w') as newFile:
        

for date, time in zip(studyDates, studyTimes):
    print(date, time)
    studyData['date'].append(date.strftime('%Y-%m-%d') if date else 'N/A')
    studyData['time'].append(time.strftime('%H:%M:%S') if time else 'N/A')

print("Script ended")