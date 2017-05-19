import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class FeaturesGenerator(torch.nn.Module):
    def __init__(self, filters, patch_size, patch_stride, pool_size, pool_stride, patch_mean=0, path_std=1):
        super(FeaturesGenerator, self).__init__()

        self.filters = filters

        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.mean = patch_mean
        self.std= path_std

        print(pool_size, pool_stride)

        self.pool = torch.nn.AvgPool2d(kernel_size=pool_size, stride=pool_stride)

    def get_patch(self, img, i_coord, j_coord):
        return np.array([img[i, j] for i in range(i_coord * self.patch_stride, i_coord * self.patch_stride + self.patch_size)
                for j in range(j_coord * self.patch_stride, j_coord * self.patch_stride + self.patch_size)]).reshape(self.patch_size, self.patch_size)

    def normalize(self, patch):
        patch -= self.mean
        patch /= self.std
        return patch

    def forward(self, inp_img):
        # print(self.filters[np.newaxis].T.shape)
        # print(inp[np.newaxis].shape)

        data_size = inp_img.shape[0]

        x_size = int((data_size - self.patch_size) / self.patch_stride) + 1
        y_size = int((data_size - self.patch_size) / self.patch_stride) + 1
        c_size = self.filters.shape[0]

        res_img = torch.FloatTensor(c_size, x_size, y_size)

        for i in range(x_size):
            for j in range(y_size):
                x = self.normalize(self.get_patch(inp_img, i, j)).flatten()
                x = np.dot(self.filters, x)
                x = Variable(torch.from_numpy(x))
                x = F.relu(x)
                res_img[:, i, j] = x.data

        pool_res = self.pool(Variable(res_img))

        return pool_res.data.numpy().flatten()