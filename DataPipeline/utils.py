from pathlib import Path
import os
import numpy as np
import shutil
import sys
import random
root_dir = Path(__file__).resolve().parent.parent
dataPath = root_dir / "physionet.org/files"
trainPath = root_dir / "data"
testPath = root_dir / "test"

# print(trainPath)
# sys.exit(1)


##Our script fills the training dataset, remove a random 30% of images from training
#move them into a test set for 70-30 split. 


#pass in sourPath that is a list of all files in a training data directory
#targetPath is a new folder

#get a list of all file paths within a folder
def getFileList(folder):
	fileList = []
	for file in os.listdir(folder):
		if (os.path.isfile(os.path.join(folder, file))):
			newFile = os.path.join(folder, file)
			fileList.append(newFile)
			
	return fileList


##for train test split do pathList = someList(randint(1, len(list)))


def transferFileAcrossDatasets(path, targetPath):
	# check if the destination folder exists and if not create it
	if not os.path.exists(targetPath):
		os.makedirs(targetPath)
	# loop over the image paths
	
	imageName = path.split(os.path.sep)[-1]
	label = path.split(os.path.sep)[-2]
	labelFolder = os.path.join(targetPath, label)
	# print(f"{imageName} {label} {labelFolder}")
	# sys.exit(1)
	
	# check to see if the label folder exists and if not create it
	if not os.path.exists(labelFolder):
		os.makedirs(labelFolder)
	# construct the destination image path and copy the current
	# image to it
	
	destination = os.path.join(labelFolder)
	#copy to target
	shutil.copy(path, destination)
	## remove file from source
	os.remove(path)
	print(f"Moved {imageName} from {path} to {destination}")
		
exampleSource = trainPath / 'NORM'
exampleTarget = testPath

# sourceList = getFileList(exampleSource)
# transferFilesAcrossDatasets(sourceList, exampleTarget)


testFile = getFileList(trainPath / 'NORM')[-1]



# fileList = [os.path.join(exampleSource, file_name) for file_name in os.listdir(exampleSource) if os.path.isfile(os.path.join(exampleSource, file_name))]
# return fileList