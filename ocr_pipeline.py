import numpy as np
import svhn_data
import spsa_clustering
import features_generator as fg
from sklearn import svm

patch_size = 9
patch_stride = 1

spsa_gamma = 1. / 6
spsa_alpha = lambda x: 0.25 / (x ** spsa_gamma)
spsa_beta = lambda x: 15. / (x ** (spsa_gamma / 4))
n_filters = 500

data_generator = svhn_data.SVHNData('/home/andrew/Projects/datasets/SVHN', patch_size, patch_stride)
data_generator.get_mean_std(10)

# data_generator.save_example()

clustering = spsa_clustering.ClusteringSPSA(n_clusters=n_filters, data_shape=patch_size * patch_size, Gammas=None,
                                            alpha=spsa_alpha, beta=spsa_beta, norm_init=False, eta=1000)

train_generator = data_generator.generate('train')
num = 0
for _ in range(data_generator.train_number):
    print(num)
    num += 1
    train_data = next(train_generator)
    for patch in train_data[0]:
        print(patch.shape)
        clustering.fit(patch)

