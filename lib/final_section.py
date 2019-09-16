import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from functools import reduce
from itertools import chain

# TODO - Combine Delayed and Final Section
class FinalSection(nn.Module):

    def __init__(self, layers, lr):
        super().__init__()
        self.layers = layers
        
        # Join layer parameters in generator function
        parameters = reduce(chain, [layer.parameters() for layer in self.layers if hasattr(layer, "parameters")])
        self.optimizer = torch.optim.Adam(parameters, lr = lr)
        # TODO add reference to cloud queue
        self.init_queues()

    def forward(self, x):
        x.requires_grad_(True)
        self.optimizer.zero_grad()
        out = self.layers[0](x)
        for layer in self.layers[1:]:
            out = layer(out)
        return out

    # TODO switch to cloud queue
    def init_queues(self):
        self.input_queue = deque()
