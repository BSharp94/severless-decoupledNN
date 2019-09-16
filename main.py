from collections import deque
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from lib.delayed_section import DelayedSection
from lib.final_section import FinalSection
from lib.master_coordinator import MasterCoordinator

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

section_1 = DelayedSection(
    [nn.Conv2d(1, 36, kernel_size = 2, padding=1, bias = False)
    , F.relu
    , nn.Conv2d(36, 36, kernel_size = 2, bias = False)
    , F.relu
    , nn.AvgPool2d(kernel_size=2)
    , nn.Dropout(p = 0.25)
    ], 0, 5e-4)

section_2 = DelayedSection(
    [nn.Conv2d(36, 64, kernel_size = 2, padding=1, bias = False)
    , F.relu
    , nn.Conv2d(64, 64, kernel_size = 2, bias = False)
    , F.relu
    , nn.AvgPool2d(kernel_size=2)
    , nn.Dropout(p = 0.25)
    , Flatten()
    ], 1, 5e-4)

section_3 = FinalSection(
    [nn.Linear(7 * 7 * 64, 1028)
    , F.relu
    , nn.Dropout(p = 0.25)
    , nn.Linear(1028, NUM_CLASSES)
    , F.softmax
    ], 5e-4)

coordinator = MasterCoordinator([
    section_1,
    section_2,
    section_3
])

for epoch_iter in range(NUM_EPOCHS):

    # Load Training Set
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=BATCH_SIZE, shuffle=True)

    print("Starting Epoch ", epoch_iter + 1)

    for batch_idx, (data, target) in enumerate(train_loader):
        
        results = coordinator.step(data, target)

        print("Batch: ", batch_idx)
        if results:
            print("\tAcc", results["acc"])




test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=BATCH_SIZE, shuffle=True)

# TODO - add testing