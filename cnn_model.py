import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid




def main():
    root_dir = Path(__file__).resolve().parent
    trainPath = root_dir / "data"
    testPath = root_dir / "test"
    
    horizontalFlipProbability = 0.40
    verticalFlipProbability = 0.40
    rotationDegrees = 20
    hFlip = transforms.RandomHorizontalFlip(horizontalFlipProbability)
    vFlip = transforms.RandomVerticalFlip(verticalFlipProbability)
    rotate = transforms.RandomRotation(degrees=rotationDegrees)


    trainDataset = ImageFolder(
        trainPath,
        transform=transforms.Compose(
            [transforms.Resize((150, 150)), hFlip, vFlip, transforms.ToTensor()]
        ),
    )
    testDataset = ImageFolder(
        testPath,
        transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor()]),
    )

    ##try experimenting batch_size = 16, when i tried it would sometimes crash due to memory issues
    batch_size = 8
    # val_size = len(testDataset)
    # train_size = len(trainDataset) - val_size

    num_workers = 4

    train_dl = DataLoader(
        trainDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_dl = DataLoader(
        testDataset, batch_size=batch_size * 2, num_workers=num_workers, pin_memory=True
    )

    
    def show_batch(dl):
        """Plot images grid of single batch"""
        for images, labels in dl:
            fig, ax = plt.subplots(figsize=(16, 12))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
            plt.show()
            break
  
    class ImageClassificationBase(nn.Module):

        def training_step(self, batch):
            images, labels = batch
            out = self(images)  # Generate predictions
            loss = F.cross_entropy(out, labels)  # Calculate loss
            return loss

        def validation_step(self, batch):
            images, labels = batch
            out = self(images)  # Generate predictions
            loss = F.cross_entropy(out, labels)  # Calculate loss
            acc = accuracy(out, labels)  # Calculate accuracy
            return {"val_loss": loss.detach(), "val_acc": acc}

        def validation_epoch_end(self, outputs):
            batch_losses = [x["val_loss"] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
            batch_accs = [x["val_acc"] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
            return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

        def epoch_end(self, epoch, result):
            print(
                "Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                    epoch, result["train_loss"], result["val_loss"], result["val_acc"]
                )
            )

    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    class EcgAnnotationClassification(ImageClassificationBase):
        ##I added some batch normalisation layers into the model to prevent that accuracy stalling we saw before
        ##Added dropout layers to prevent overfitting
        '''train_loss and val_loss should be similar, if train_loss >> val_loss
        that indicates that the model performs much better on the training data than testing data, 
        indicating overfitting. The opposite would indicate underfitting,'''
        def __init__(self):

            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                ##Added normalisation functions
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2, 2),
                #Dropout Layers
                nn.Dropout(0.25),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.50),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.25),
                nn.Flatten(),
                nn.Linear(82944, 1024),
                nn.ReLU(),
                
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(
                    512, 10
                ),  # this is the output size need to change to match the input tensor
            )

        def forward(self, xb):
            return self.network(xb)

    model = EcgAnnotationClassification()
    # for images, labels in train_dl:
    #     print("images.shape:", images.shape)
    #     out = model(images)
    #     print("out.shape:", out.shape)
    #     print("out[0]:", out[0])
    #     break

    def get_default_device():
        """Set Device to GPU or CPU"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def to_device(data, device):
        "Move data to the device"
        if isinstance(data, (list, tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    class DeviceDataLoader:
        """Wrap a dataloader to move data to a device"""

        def __init__(self, dl, device):
            self.dl = dl
            self.device = device

        def __iter__(self):
            """Yield a batch of data after moving it to device"""
            for b in self.dl:
                yield to_device(b, self.device)

        def __len__(self):
            """Number of batches"""
            return len(self.dl)

    device = get_default_device()

    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    model = to_device(EcgAnnotationClassification(), device)

    @torch.no_grad()
    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)

    def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
        print('Beginning training...')
        print(f"N = {len(testDataset) + len(trainDataset)}")
        print(f"Number of training samples: {len(trainDataset)}")
        print(f"Number of testing samples: {len(testDataset)}\n")
        print(f"Number of SCP categories: {len(trainDataset.classes)}")
        print(f"Number of training batches passed to the model: {len(train_dl)} (should correspond to number of training samples/batch_size)")
        print(f"Number of testing batches passed to model: {len(val_dl)} (should correspond to number of testing samples/ 2*batch_size)")
        history = []
        
        optimizer = opt_func(model.parameters(), lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        stepped = False
        for epoch in range(epochs):
            model.train()
            train_losses = []
            for batch in train_loader:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            result = evaluate(model, val_loader)
            result["train_loss"] = torch.stack(train_losses).mean().item()
            model.epoch_end(epoch, result)
            history.append(result)
            scheduler.step(result["val_loss"])
            new_lr = scheduler.optimizer.param_groups[0]['lr']
            if((lr != new_lr) & (stepped == False)):
                print('Lr stepdown due to val_acc plateau!')
                stepped = True
            print(f"lr: {new_lr}\n")

        return history

    # initial evaluation of the model
    evaluate(model, val_dl)

    # Train the model
    num_epochs = 30
    opt_func = torch.optim.Adam
    #lr = 0.0001 with a stepdown has been working best for samplesize = 600. 
    lr = 0.0001
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    # Plotting functions
    def plot_accuracies(history):
        accuracies = [x["val_acc"] for x in history]
        plt.plot(accuracies, "-x")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title("Accuracy vs. No. of epochs")
        plt.show();

    def plot_losses(history):
        train_losses = [x.get("train_loss") for x in history]
        val_losses = [x["val_loss"] for x in history]
        plt.plot(train_losses, "-bx")
        plt.plot(val_losses, "-rx")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(["Training", "Validation"])
        plt.title("Loss vs. No. of epochs")
        plt.show()

    plot_accuracies(history)
    plot_losses(history)

    test_loader = DeviceDataLoader(DataLoader(testDataset, batch_size * 2), device)
    result = evaluate(model, test_loader)
    print(result)

    # Save the model
    torch.save(model.state_dict(), "ecg-annotation.pth")


if __name__ == "__main__":
    main()