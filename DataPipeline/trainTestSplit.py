
from utils import getFileList, transferFileAcrossDatasets
from pathlib import Path
import os
import sys
import random
#i'll clean this up to variable imports when I get file permissisons
classes = [
    "NORM",
    "IMI",
    "NDT",
    "ASMI",
    " LVH",
    "LAFB",
    "IRBBB",
    "CLBBB",
    "NST_",
    "CRBBB",
]

root_dir = Path(__file__).resolve().parent.parent

dataPath = root_dir / 'SampleData'
trainPath = root_dir / "data"
testPath = root_dir / "test"

classes = [
    "NORM",
    "IMI",
    "NDT",
    "ASMI",
    " LVH",
    "LAFB",
    "IRBBB",
    "CLBBB",
    "NST_",
    "CRBBB",
]
# Define the directory containing the patient folders
root_dir = Path(__file__).resolve().parent.parent

dataPath = root_dir / 'SampleData'
trainPath = root_dir / "data"
testPath = root_dir / "test"

# make train/test folder for each of the 10 SCP codes

def makeTestFolders(testPath, classes):
    for label in classes:
        newTestPath = os.path.join(testPath,label.strip())
        if not (os.path.exists(newTestPath)):
            os.makedirs(newTestPath)
            print(f"New Directories: {newTestPath}")

makeTestFolders(testPath, classes)

#training ratio is a decimal, 0.70 represent a 70/30 train test split

def trainTestSplit (trainPath, testPath, trainingRatio):
    for label in classes:
        fileList = getFileList(os.path.join(trainPath, label.strip()))
        if (fileList):
            #this is scuffed af 
            index = len(fileList)
            for file in fileList:
                try:
                    if((random.randint(1,100) >= trainingRatio * 100)):
                        transferFileAcrossDatasets(fileList[random.randint(0, index)], testPath)
                        index -= 1
                        print('transferred')
                    else:
                        print('no transfer')
                except Exception as e:
                    print('File not found')
                    continue
        
#Running this again will mess up current layout
trainTestSplit(trainPath, testPath, 0.80)