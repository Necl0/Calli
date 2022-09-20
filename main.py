import torch
import torch.nn as nn
import torchvision
from torch.nn import Sequential, Linear, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pickle

from pydantic import BaseModel, conint, confloat, validator, ValidationError
from typing import Optional, Any

from torchvision.datasets import KMNIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Hyperparameters(BaseModel):
    """Hyperparameter class"""
    num_epochs: Optional[conint(gt=0)] = 20
    num_classes: Optional[conint(gt=0)] = 10
    batch_size: Optional[conint(gt=0)] = 32
    learning_rate: Optional[confloat(gt=0.0)] = 0.01
    
    @validator('batch_size')
    def power_of_two(cls, v: int) -> int:
        assert (v & (v-1) == 0) and v != 0, 'Batch size should be a power of two.'
        return v

try:
    m1: Hyperparameters = Hyperparameters()
except ValidationError as e:
    print(e)


# load KMNIST
train_data: KMNIST = torchvision.datasets.KMNIST(root='../../data/',
                                         train=True,
                                         transform=transforms.ToTensor(),
                                         download=True)

test_data: KMNIST = torchvision.datasets.KMNIST(root='../../data/',
                                        train=False,
                                        transform=transforms.ToTensor())

# Data Loader
train_loader: DataLoader[Any] = DataLoader(dataset=train_data,
                          batch_size=m1.batch_size,
                          shuffle=True)

test_loader: DataLoader[Any] = DataLoader(dataset=test_data,
                         batch_size=m1.batch_size,
                         shuffle=False)


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1: Sequential = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2: Sequential = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc: Linear = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out: object = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


model: ConvNet = ConvNet(m1.num_classes).to(device)

# Loss and optimizer
criterion: CrossEntropyLoss = nn.CrossEntropyLoss()
optimizer: Adam = torch.optim.Adam(model.parameters(), lr=m1.learning_rate)

# Train the model
total_step: int = len(train_loader)
for epoch in range(m1.num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images: object = images.to(device)
        labels: object = labels.to(device)

        # Forward pass
        outputs: object = model(images)
        loss: object = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{m1.num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item()}')

# Test the model
model.eval()  # eval mode (batch norm uses moving mean/variance instead of mini-batch mean/variance)

with torch.no_grad():
    correct: int = 0
    total: int = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy of the model on the 10000 test images: {100 * correct / total}')

with open('model.pkl', 'wb') as files:
    pickle.dump(model, files)
