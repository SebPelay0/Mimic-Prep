
import wfdb 

from pathlib import Path
import pandas as pd
path = r"/home/sebastian/VIP/Mimic-Prep/SampleData/p10000032/s44458630/44458630"
studyNum = path.split('/')[-1]  #get last study num
graphTitle = 'Study' + studyNum

#find all header files within this folder
folderNum = 'p10000032'
paths = list(Path(folderNum).rglob("*.hea"))
studyData = {'studyNum':[],'date':[],'time':[]}

i = 0
for file in paths:
    #Extract study number
    pathObject = paths[i]
    studyName = pathObject.name.split('.hea')[0]

    print(studyName)
    study = file.stem
    metadata = wfdb.rdheader(f'{file.parent}/{file.stem}')
    
    studyData['date'].append(metadata.base_date)
    studyData['time'].append(metadata.base_time)
    i = i + 1



# rd_record = wfdb.rdrecord(path) 
# print(rd_record)
# wfdb.plot_wfdb(record=rd_record, figsize=(24,18), title=graph_title, ecg_grids='all')


# folderNum = 'p10000032'
# paths = list(Path(folderNum).rglob("*.hea"))