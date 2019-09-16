import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from functools import reduce
from itertools import chain

class DelayedSection(nn.Module):

    def __init__(self, layers, section_index, lr):
        super().__init__()
        self.layers = layers
        self.section_index = section_index
        
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
        self.output_queue.append(out)
        return out

    def backwards_updates(self):
        if self.gradient_queue:
            self.optimizer.zero_grad()
            gradient = self.gradient_queue.popleft()
            output = self.output_queue.popleft()
            output.backward(gradient)
            self.optimizer.step()

            if self.section_index > 0:
                input_signal = self.init_queues().popleft()
                return input_signal.grad.detach().clone()

    # TODO switch to cloud queue
    def init_queues(self):
        self.gradient_queue = deque()
        self.output_queue = deque()
        if self.section_index > 0:
            self.input_queue = deque()
