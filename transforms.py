import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import sys

##idk i can't import these from PTBPipeline.py I think I dont have permissions?
root_dir = Path(__file__).resolve().parent

dataPath = root_dir / 'SampleData'
trainPath = root_dir / "data"
testPath = root_dir / "test"




#experiment with parameters for accuracy. 
INPUT_HEIGHT = 128
INPUT_WIDTH = 128
BATCH_SIZE = 8

#define probability values on how likely it is that each image is flipped, to introduce more variety into dataset
horizontalFlipProbability = 0.25
verticalFlipProbability = 0.25
rotationDegrees = 15


# initialize our image transformations...

resize = transforms.Resize(size=(INPUT_HEIGHT,
        INPUT_WIDTH))
hFlip = transforms.RandomHorizontalFlip(horizontalFlipProbability)
vFlip = transforms.RandomVerticalFlip(verticalFlipProbability)
rotate = transforms.RandomRotation(degrees=rotationDegrees)


#pass all transformation args into a single transformation function
#also converts input images to pytorch tensors. This would be the same for raw signal data/numpy arrays.
trainTransforms = transforms.Compose([resize, hFlip, vFlip, rotate, transforms.ToTensor()])

##in example code they call this valTransforms, idk if we wanna change it back?
testTransforms = transforms.Compose([resize, transforms.ToTensor()])

##CREATE TRAINING AND TESTING DATASET FROM SORTED DIRECTORIES

trainingDataset = ImageFolder(root=trainPath, transform=trainTransforms)
testingDataset = ImageFolder(root=testPath, transform=testTransforms)
##these datasets are lists of tuples, can simply be accessed via trainingDataset[i]
##each entry is a tuple of (value, label)
## value is the actual signal image and label is the SCP code with the highest confidence rating

print("[INFO] training dataset contains {} samples...".format(
        len(trainingDataset)))
print("[INFO] validation dataset contains {} samples...".format(
        len(testingDataset)))

print("[INFO] creating training and validation set dataloaders...")

#Our batch size determines how many image samples are sent to the model at a time
trainDataLoader = DataLoader(trainingDataset, batch_size=BATCH_SIZE, shuffle=True)
valDataLoader = DataLoader(testingDataset, batch_size=BATCH_SIZE)

##from this point implement those batch visualisation functions? 