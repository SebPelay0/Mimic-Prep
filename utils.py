from pathlib import Path
import os
import numpy as np
import shutil
import sys
root_dir = Path(__file__).resolve().parent
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



#takes a list of all file paths within a given folder, copies to target then removes from original folder

#to move into test directories, targetPath =>  testPath = root_dir / "test"
def transferFilesAcrossDatasets(pathList, targetPath):
	# check if the destination folder exists and if not create it
	if not os.path.exists(targetPath):
		os.makedirs(targetPath)
	# loop over the image paths
	for path in pathList:
		# grab image name and its label from the path and create
		# a placeholder corresponding to the separate label folder
		
		imageName = path.split(os.path.sep)[-1]
		label = path.split(os.path.sep)[-2]
		labelFolder = os.path.join(targetPath, label)
		print(path)
		print(f"{imageName} {label} {labelFolder}")
		sys.exit(1)
		# check to see if the label folder exists and if not create it
		if not os.path.exists(labelFolder):
			os.makedirs(labelFolder)
		# construct the destination image path and copy the current
		# image to it
		destination = os.path.join(labelFolder)
		shutil.copy(path, destination)
		
exampleSource = trainPath / 'NORM'
exampleTarget = testPath

sourceList = getFileList(exampleSource)
transferFilesAcrossDatasets(sourceList, exampleTarget)

#transferFilesAcrossDatasets(sourcePath, exampleTarget)



# fileList = [os.path.join(exampleSource, file_name) for file_name in os.listdir(exampleSource) if os.path.isfile(os.path.join(exampleSource, file_name))]
# return fileList