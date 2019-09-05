from collections import deque
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
        self.conv1 = nn.Conv2d(1, 36, kernel_size = 2, padding=1, bias = False)
        self.conv2 = nn.Conv2d(36, 36, kernel_size = 2, bias = False)
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
        self.conv3 = nn.Conv2d(36, 64, kernel_size = 2, padding = 1, bias = False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size = 2, bias = False)
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
        self.dropout3 = nn.Dropout(p = 0.6)
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

section1_pd_queue = deque()
section2_pd_queue = deque()

section2_input_queue = deque()
section3_input_queue = deque()

section1_gradient_queue = deque()
section2_gradient_queue = deque()

label_queue = deque()

lr_shrink = 0.2
for epoch_iter in range(NUM_EPOCHS):

    print("Starting Epoch ", epoch_iter + 1)


    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=BATCH_SIZE, shuffle=True)

    for batch_idx, (data, target) in enumerate(train_loader):

        #! For now, let skip the last iteration
        if (data.shape[0] < BATCH_SIZE):
            continue

        # Section 1 Backwards
        model1.train()
        optimizer1.zero_grad()
        if section1_gradient_queue and section1_pd_queue:
            back_grad = section1_gradient_queue.popleft()
            pd = section1_pd_queue.popleft()
            pd.backward(lr_shrink * back_grad)
            optimizer1.step()
            optimizer1.zero_grad()

        # Section 1 Forward
        data.requires_grad_(True) # Allow gradient to be tracked
        model1.zero_grad()
        optimizer1.zero_grad()
        output1 = model1.forward(data) # pass output forward
        section1_pd_queue.append(output1) # remember past data

        # Section 2 Backward
        model2.train()
        optimizer2.zero_grad()
        if section2_gradient_queue and section2_pd_queue:
            back_grad = section2_gradient_queue.popleft()
            pd = section2_pd_queue.popleft()
            original_input = section2_input_queue.popleft()
            pd.backward(lr_shrink *  back_grad)
            optimizer2.step()
            optimizer1.zero_grad()
            section1_gradient_queue.append(original_input.grad.detach().clone())

        # Section 2 Forward
        if section2_input_queue:
            output2_input = section2_input_queue[len(section2_input_queue) - 1]
            output2_input.requires_grad_(True)
            output2 = model2.forward(output2_input)
            
            section2_pd_queue.append(output2) # remember past data
            optimizer2.zero_grad()

        # Section 3 Forward and backwards
        if section3_input_queue:
            output3_input = section3_input_queue.popleft()
            output3_input.requires_grad_(True)
            
            model3.train()
            output3 = model3(output3_input)

            # Calculate Loss And Accuracy
            target_true =  label_queue.popleft()
            loss = F.cross_entropy(output3,target_true)
            _, predicted = torch.max(output3.data, 1)
            print(predicted)
            correct = (predicted == target_true).sum().item()
            acc = correct / target_true.size(0)
            print("Epoch: ", epoch_iter, " Iter: ", batch_idx, " Loss: ", loss, " Acc: ", acc)

            # Section 3 backward
            optimizer3.zero_grad()
            loss.backward()

            optimizer3.step()
            section2_gradient_queue.append(output3_input.grad.detach().clone())
        else:
            print("***** No Output Data yet")

        section2_input_queue.append(output1.detach().clone())

        if 'output2' in locals():
            section3_input_queue.append(output2.detach().clone())

        label_queue.append(target)

print("complete")