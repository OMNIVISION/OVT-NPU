import os

import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.init as init


# MNist untrained model
class MNist(nn.Module):
    def __init__(self, folding=False, no_standardize=False):
        super().__init__()

        # Create layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2)
        self.l1 = nn.Linear(4 * 4 * 32, 128)
        self.l2 = nn.Linear(128, 10)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize convolutional layers with Kaiming initialization
        init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        # init.constant_(self.conv1.bias, 0)

        init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
        init.constant_(self.conv2.bias, 0)

        # Initialize linear layers
        init.kaiming_normal_(self.l1.weight, mode="fan_in", nonlinearity="relu")
        init.constant_(self.l1.bias, 0)

        init.xavier_uniform_(self.l2.weight)
        init.constant_(self.l2.bias, 0)

    def forward(self, x):
        h = self.pool(self.act(self.conv1(x)))
        h = self.pool(self.act(self.conv2(h)))
        h = self.pool(h)
        h = h.flatten(start_dim=1)
        h = self.act(self.l1(h))
        # h = self.act(self.l2(h))
        h = self.l2(h)
        return nn.functional.softmax(h, dim=1)
