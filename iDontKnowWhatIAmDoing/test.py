
import wfdb 

from pathlib import Path
import pandas as pd
import datetime


folderNum = 'p10000032'
dataPath = r"/home/sebastian/VIP/Mimic-Prep/SampleData/"

paths = list(Path(folderNum).rglob("*.hea"))
studyData = {'studyNum':[],'date':[],'time':[]}
studyDates = []
studyTimes = []
i = 0

for file in paths:
    
   
    
    pathObject = paths[i]
    #Extract study number
    studyNum = pathObject.name.split('.hea')[0]
    studyData['studyNum'].append(studyNum)

    #Extract date and times
    study = file.stem
    metadata = wfdb.rdheader(f'{file.parent}/{file.stem}')
    studyDates.append(metadata.base_date)
    studyTimes.append(metadata.base_time)


    #get file path to produce image
    filePath = (dataPath + str(pathObject)).split('.hea')[0]
    rd_record = wfdb.rdrecord(filePath) 

    fig = wfdb.plot_wfdb(record=rd_record, figsize=(24,18), title=studyNum, ecg_grids='all', return_fig=True)
    fig.savefig(studyNum + ".png")


    break
    

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