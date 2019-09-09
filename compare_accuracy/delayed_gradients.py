from collections import deque
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

DATA_DIR  = "data"
NUM_CLASSES = 10
NUM_CHANNELS = 10
NUM_EPOCHS = 10
BATCH_SIZE = 64

VALIDATION_ITER = 100

# Custom Flatten Module
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# Alexnet Model
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
        self.dropout3 = nn.Dropout(p = 0.25)
        self.linear2 = nn.Linear(1028, NUM_CLASSES)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.dropout3(out)
        out = F.softmax(self.linear2(out))

        return out

# Section 1 model, optimizers, and queue
model1 = Section1()
optimizer1 = torch.optim.Adam(model1.parameters(), lr = 5e-4)
model1_output_queue = deque()
model1_gradient_queue = deque()

# Section 2 model, optimizers, and queue
model2 = Section2()
optimizer2 = torch.optim.Adam(model2.parameters(), lr = 5e-4)
model2_input_queue = deque()
model2_output_queue = deque()
model2_gradient_queue = deque()

# Section 3 model, optimizers, and queue
model3 = Section3()
optimizer3 = torch.optim.Adam(model3.parameters(), lr = 5e-4)
model3_input_queue = deque()
model3_label_queue = deque()

model3_label_epoch = 0
model3_label_iteration = 0

NUM_BLANKS_DATA_ITERATIONS = 4 # TODO - Make an algorithm based on number of sections

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=BATCH_SIZE, shuffle=True)

validation_records = []
test_records = []

lr_shrink = 1

for epoch_iter in range(NUM_EPOCHS):

    if epoch_iter == 3:
        lr_shrink = 0.5

    if epoch_iter == 6:
        lr_shrink = 0.2

    if epoch_iter == 8:
        lr_shrink = 0.1

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=BATCH_SIZE, shuffle=True)

    print("Starting Epoch ", epoch_iter + 1)

    validation_epoch_record = []

    # Train
    # TODO - add four more iterations to allow signal to finish propogation
    for batch_idx, (data, target) in enumerate(train_loader):
        
        validation = batch_idx % VALIDATION_ITER == 0

        if validation:
            model1.eval()
            model2.eval()
            model3.eval()

        # Section 1 Backwards
        if not validation and model1_gradient_queue and model1_output_queue:
            model1.train()
            optimizer1.zero_grad()
            gradient = model1_gradient_queue.popleft()
            output = model1_output_queue.popleft()
            output.backward(lr_shrink * gradient)
            optimizer1.step()

        # Section 1 Forwards
        data.requires_grad_(True) 
        optimizer1.zero_grad()
        output = model1(data)
        model1_output_queue.append(output)

        # Section 2 Backwards
        if not validation and model2_gradient_queue and model2_output_queue and model2_input_queue:
            model2.train()
            optimizer2.zero_grad()
            gradient = model2_gradient_queue.popleft()
            output = model2_output_queue.popleft()
            input_signal = model2_input_queue.popleft()
            output.backward(lr_shrink * gradient)
            optimizer2.step()
            model1_gradient_queue.append(input_signal.grad.detach().clone())

        # Section 2 Forward
        if model2_input_queue:
            input_signal = model2_input_queue[len(model2_input_queue) - 1]
            input_signal.requires_grad_(True)
            output = model2(input_signal)
            model2_output_queue.append(output)

        # Section3 Forwards and Backwards
        if model3_input_queue and model3_label_queue:
            input_signal = model3_input_queue.popleft()
            input_signal.requires_grad_(True)
            optimizer3.zero_grad()
            output = model3(input_signal)

            label = model3_label_queue.popleft()
            loss = F.cross_entropy(output,label)
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == label).sum().item()
            acc = correct / label.size(0)
            print("Epoch: ", epoch_iter, " Iter: ", batch_idx, " Loss: ", loss, " Acc: ", acc)

            # check for updates in validation records.
            if model3_label_iteration >= len(train_loader) - 1:
                model3_label_epoch = model3_label_epoch + 1
                model3_label_iteration = 0

            if validation:
                validation_epoch_record.append({"Epoch": model3_label_epoch, "Batch": model3_label_iteration, "Accuracy": acc})
            model3_label_iteration = model3_label_iteration + 1

            loss.backward()
            optimizer3.step()
            model2_gradient_queue.append(input_signal.grad.detach().clone())

        # Record labels
        model3_label_queue.append(target)

        # Forward pass
        if model1_output_queue:
            model2_input_queue.append(model1_output_queue[len(model1_output_queue) - 1].detach().clone())

        if model2_output_queue:
            model3_input_queue.append(model2_output_queue[len(model2_output_queue) - 1].detach().clone())

    validation_records.append(validation_epoch_record)

    # Test
    correct = 0
    for data, target in test_loader:
        output = model1(data)
        output = model2(output)
        output = model3(output)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    acc = 100. * correct / len(test_loader.dataset)
    print('Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), acc))
    test_records.append({"Epoch": epoch_iter, "Accuracy": acc})

np.save("delayed_model_validation_lr_shrink_downscale", validation_records)
np.save("delayed_model_test_lr_shrink_downscale", test_records)

print("complete")