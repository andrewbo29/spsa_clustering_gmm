import torch
import torch.nn.functional as F
import numpy as np


class FeaturesGenerator(torch.nn.Module):
    def __init__(self, filters, pool_size, pool_stride):
        super(FeaturesGenerator, self).__init__()

        self.filters = filters

        self.pool = torch.nn.AvgPool2d(pool_size, pool_stride)

    def forward(self, inp):
        x = np.dot(self.filters, inp)
        x = F.relu(x)
        x = self.pool(x)
        return x