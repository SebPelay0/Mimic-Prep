
import wfdb 
import numpy
path = "/home/sebastian/VIP/Mimic-Prep/SampleData/p10000032/s44458630/44458630"
    
rd_record = wfdb.rdrecord(path) 
print(rd_record)
wfdb.plot_wfdb(record=rd_record, figsize=(24,18), title='Study 41420867 example', ecg_grids='all')