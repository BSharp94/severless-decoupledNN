import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

BATCH_SIZE = 64

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=BATCH_SIZE, shuffle=True)

# Get single batch of data
data_batch = None
for batch_idx, (data, target) in enumerate(train_loader):
    data_batch = data
    break


# convert to numpy
numpy_data = data_batch.data.numpy()
byte_data = numpy_data.tobytes()

#! Make sure to add the dtype as float32, otherwise its float64
decoded = np.frombuffer(byte_data, dtype=np.float32) 



