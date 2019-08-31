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

# Section 1
class Section1(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 36, kernel_size = 2, padding=1)
        self.conv2 = nn.Conv2d(36, 36, kernel_size = 2)
        self.pooling1 = nn.AvgPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(p = 0.25)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.pooling1(out)
        out = self.dropout1(out)

        return out

class Section2(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv3 = nn.Conv2d(36, 64, kernel_size = 2, padding = 1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size = 2)
        self.pooling2 = nn.AvgPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(p = 0.25)
        self.flatten = Flatten()

    def forward(self, x):
        out = F.relu(self.conv3(x))
        out = F.relu(self.conv4(out))
        out = self.pooling2(out)
        out = self.dropout2(out)
        out = self.flatten(out)

        return out

class Section3(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(7 * 7 * 64, 1028)
        self.dropout3 = nn.Dropout(p = 0.25)
        self.linear2 = nn.Linear(1028, NUM_CLASSES)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.dropout3(out)
        out = F.softmax(self.linear2(out))

        return out

model1 = Section1()
model2 = Section2()
model3 = Section3()

criterion = nn.CrossEntropyLoss()

optimizer1 = torch.optim.Adam(model1.parameters(), lr = 5e-4)
optimizer2 = torch.optim.Adam(model2.parameters(), lr = 5e-4)
optimizer3 = torch.optim.Adam(model3.parameters(), lr = 5e-4)

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
        
        data.requires_grad_(True)

        model1.train()
        model1.zero_grad()
        # backwards
        optimizer1.step()
        output1 = model1(data)
        model1.zero_grad()


        model2.train()
        model2.zero_grad()
        # backwards
        optimizer2.step()
        output2_input = output1.detach().clone()
        output2_input.requires_grad_(True)
        output2 = model2(output2_input)
        model2.zero_grad()

        model3.train()
        model3.zero_grad()
        optimizer3.step()
        output3_input = output2.detach().clone()
        output3_input.requires_grad_(True)
        output3 = model3(output3_input)

        loss = F.cross_entropy(output3, target)
        _, predicted = torch.max(output3.data, 1)
        correct = (predicted == target).sum().item()
        acc = correct / target.size(0)

        model3.zero_grad()

        loss.backward()
        output2.backward(output3_input.grad)
        output1.backward(output2_input.grad)

        optimizer3.step()
        optimizer2.step()
        optimizer1.step()
        

        
        # calc accuracy
        _, indices = output3.max(1)
        acc = target.eq(indices).sum().item() / BATCH_SIZE
        print("Acc: ", acc)

print("complete")