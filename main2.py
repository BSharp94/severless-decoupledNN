import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

DATA_DIR  = "data"
NUM_CLASSES = 10
NUM_CHANNELS = 1
NUM_EPOCHS = 10
BATCH_SIZE = 64

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class AlexModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 36, kernel_size = 2, padding=1)
        self.conv2 = nn.Conv2d(36, 36, kernel_size = 2)
        self.pooling1 = nn.AvgPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(p = 0.25)

        self.conv3 = nn.Conv2d(36, 64, kernel_size = 2, padding = 1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size = 2)
        self.pooling2 = nn.AvgPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(p = 0.25)

        # Add Flatten in forward and backwards
        self.flatten = Flatten()
        self.linear1 = nn.Linear(7 * 7 * 64, 1028)
        self.dropout3 = nn.Dropout(p = 0.25)
        self.linear2 = nn.Linear(1028, NUM_CLASSES)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.pooling1(out)
        out = self.dropout1(out)

        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.pooling2(out)
        out = self.dropout2(out)

        out = self.flatten(out)
        out = F.relu(self.linear1(out))
        out = self.dropout3(out)
        out = F.softmax(self.linear2(out))

        return out

model = AlexModel()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=BATCH_SIZE, shuffle=True)

for epoch_iter in range(NUM_EPOCHS):

    print("starting epoch ", epoch_iter + 1)

    # train?
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        out = model.forward(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        
        # calc accuracy
        _, indices = out.max(1)
        acc = target.eq(indices).sum().item() / BATCH_SIZE
        print("Acc: ", acc)

print("complete")