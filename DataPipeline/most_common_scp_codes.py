#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import csv
from collections import defaultdict
import re

# Define the directory containing the patient folders
root_dir = Path(__file__).resolve().parent.parent
dataPath = root_dir / "physionet.org/files/ptb-xl/1.0.3"

# Use rglob to find the main CSV file
csv_files = list(dataPath.rglob("ptbxl_database.csv"))[0]

# Initialise the dictionary that will contain the scp code and counts
scp_count = defaultdict(int)

with open(csv_files) as file_obj:
    reader_obj = csv.reader(file_obj)
    headers = next(reader_obj)  # Read the header row
    scp_codes_index = headers.index("scp_codes")  # Get the index of the scp_codes

    for row in reader_obj:
        scp_code = row[scp_codes_index]  # Extract the scp_code
        # Extract the highest confidence scp_code itself
        scp_code = re.findall(r"(\w+)':\s*(\d+\.\d+)", scp_code)
        useful_scp_code = max(scp_code, key=lambda x: float(x[1]))
        highest_scp_code, value = useful_scp_code
        scp_count[highest_scp_code] += 1

# Sort the dictionary
top_10_scp_codes = sorted(scp_count.items(), key=lambda item: item[1], reverse=True)[
    :10
]

print("Top 10 most common scp_codes:")
for scp_code, count in top_10_scp_codes:
    print(f"{scp_code}: {count}")
