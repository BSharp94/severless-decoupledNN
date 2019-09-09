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

class AlexnetModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 36, kernel_size = 2, padding=1, bias = False)
        self.conv2 = nn.Conv2d(36, 36, kernel_size = 2, bias = False)
        self.pooling1 = nn.AvgPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(p = 0.25)

        self.conv3 = nn.Conv2d(36, 64, kernel_size = 2, padding = 1, bias = False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size = 2, bias = False)
        self.pooling2 = nn.AvgPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(p = 0.25)
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

model = AlexnetModel()
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

validation_records = []
test_records = []
for epoch_iter in range(NUM_EPOCHS):

    print("Starting Epoch ", epoch_iter + 1)

    validation_epoch_record = []
    # Train
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # Forward
        optimizer.zero_grad()
        output = model(data)

        loss = F.cross_entropy(output, target)
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == target).sum().item()
        acc = correct / target.size(0)
        print("Epoch: ", epoch_iter, " Iter: ", batch_idx, " Loss: ", loss, " Acc: ", acc)

        if batch_idx % VALIDATION_ITER == 0:
            validation_epoch_record.append({"Epoch": epoch_iter, "Batch": batch_idx, "Accuracy": acc})
        else:
            # Backwards
            loss.backward()
            optimizer.step()

    # Record Validation
    validation_records.append(validation_epoch_record)

    # Test
    model.eval()
    correct = 0
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    acc = 100. * correct / len(test_loader.dataset)
    print('Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), acc))
    test_records.append({"Epoch": epoch_iter, "Accuracy": acc})


# Save the output
np.save("control_validation", validation_records)
np.save("control_test", test_records)

print("complete")