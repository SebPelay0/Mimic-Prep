
import wfdb 
import numpy
path = "/home/sebastian/VIP/Mimic-Prep/SampleData/p10000032/s44458630/44458630"
study_num = path.split('/')[-1]  #get last study num
graph_title = 'Study' + study_num


print(study_num)
rd_record = wfdb.rdrecord(path) 
print(rd_record)
wfdb.plot_wfdb(record=rd_record, figsize=(24,18), title=graph_title, ecg_grids='all')