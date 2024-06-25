
import wfdb 

from pathlib import Path
import pandas as pd
import datetime

path = r"/home/sebastian/VIP/Mimic-Prep/SampleData/p10000032/s44458630/44458630"
studyNum = path.split('/')[-1]  #get last study num
graphTitle = 'Study' + studyNum

#find all header files within this folder
folderNum = 'p10000032'
paths = list(Path(folderNum).rglob("*.hea"))
studyData = {'studyNum':[],'date':[],'time':[]}
studyDates = []
studyTimes = []
i = 0
for file in paths:
    #Extract study number
    pathObject = paths[i]
    studyNum = pathObject.name.split('.hea')[0]
    studyData['studyNum'].append(studyNum)

    print(studyNum)
    study = file.stem
    metadata = wfdb.rdheader(f'{file.parent}/{file.stem}')
    
    studyDates.append(metadata.base_date)
    studyTimes.append(metadata.base_time)

    i = i + 1
j = 0
for study in studyDates:
    studyData['date'].append(studyDates[j].strftime('%Y-%m-%d'))
    studyData['time'].append(studyTimes[j].strftime('%Y-%m-%d'))
    j =j + 1


    

print(studyData)


# rd_record = wfdb.rdrecord(path) 
# print(rd_record)
# wfdb.plot_wfdb(record=rd_record, figsize=(24,18), title=graph_title, ecg_grids='all')


# folderNum = 'p10000032'
# paths = list(Path(folderNum).rglob("*.hea"))