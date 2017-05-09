import numpy as np
from skimage import color
import torchvision.datasets as dset
from PIL import Image
import os.path


class SVHNData(object):

    def __init__(self, root, patch_size, patch_stride):
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.root = root

        self.train = dset.SVHN(root, split='train', transform=None, target_transform=None, download=False)
        self.data_size = self.train.data.shape[2]
        self.train_number = self.train.data.shape[0]

        self.train_gray = np.zeros((self.train.data.shape[0], self.data_size, self.data_size))
        self.train_labels = np.zeros(self.train.data.shape[0])
        for i in range(self.train.data.shape[0]):
            self.train_gray[i] = color.rgb2gray(np.transpose(self.train.data[i], (1, 2, 0)))
            self.train_labels[i] = self.train.labels[i][0]

        self.test = dset.SVHN(root, split='test', transform=None, target_transform=None, download=False)

        self.test_gray = np.zeros((self.test.data.shape[0], self.data_size, self.data_size))
        self.test_labels = np.zeros(self.test.data.shape[0])
        self.test_number = self.test.data.shape[0]
        for i in range(self.test.data.shape[0]):
            self.test_gray[i] = color.rgb2gray(np.transpose(self.test.data[i], (1, 2, 0)))
            self.test_labels[i] = self.test.labels[i][0]

        self.mean = np.zeros(self.train.data.shape[1])
        self.std = np.ones(self.train.data.shape[1])

    def get_patch(self, img, i_coord, j_coord):
        return np.array([img[i, j] for i in range(i_coord * self.patch_stride, i_coord * self.patch_stride + self.patch_size)
                for j in range(j_coord * self.patch_stride, j_coord * self.patch_stride + self.patch_size)]).reshape(self.patch_size, self.patch_size)

    def get_img_patches(self, img):
        x = [self.get_patch(img, i, j) for i in range(int((self.data_size - self.patch_size) / self.patch_stride) + 1)
             for j in range(int((self.data_size - self.patch_size) / self.patch_stride) + 1)]
        np.random.shuffle(x)
        return x

    def get_mean_std(self, number):
        patches = np.zeros((number * (int((self.data_size - self.patch_size) / self.patch_stride) + 1) * (int((self.data_size - self.patch_size) / self.patch_stride) + 1),
                            self.patch_size, self.patch_size))

        ind = np.random.choice(self.train.data.shape[0], size=number, replace=False)
        j = 0
        for i in ind:
            img_pathes = self.get_img_patches(self.train_gray[i])
            for k in range(len(img_pathes)):
                patches[j] = img_pathes[k]
                j += 1
        # patches = np.array(patches)

        # p_mean = patches.mean(0)
        # self.mean = (p_mean[0, :].mean(), p_mean[1, :].mean())
        # p_std = patches.std(0)
        # self.std = (p_std[0, :].std(), p_std[1, :].std())

        self.mean = patches.mean(0)
        self.std = patches.std(0)

    def normalize(self, patch):
        patch -= self.mean
        patch /= self.std
        return patch

    def generate(self, split):
        if split == 'train':
            for i in range(self.train.data.shape[0]):
                yield ([self.normalize(patch) for patch in self.get_img_patches(self.train_gray[i])], self.train_labels[i])
        elif split == 'test':
            for i in range(self.test.data.shape[0]):
                yield [self.normalize(patch) for patch in self.get_img_patches(self.test_gray[i])], self.test_labels[i]

    def save_example(self):
        ind = np.random.randint(0, self.train.data.shape[0])
        i = 0
        for patch in self.get_img_patches(self.train_gray[ind]):
            patch = (((patch - patch.min()) / (patch.max() - patch.min())) * 255.9).astype(np.uint8)
            img = Image.fromarray(patch)
            img.save(os.path.join(self.root, 'tmp', '{0}.png'.format(i)))
            i += 1
        img = self.train_gray[ind]
        img = (((img - img.min()) / (img.max() - img.min())) * 255.9).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(self.root, 'tmp', 'image.png'))



