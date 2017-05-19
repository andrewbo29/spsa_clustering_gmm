import numpy as np
import svhn_data
import spsa_clustering
import features_generator as fg
from sklearn import svm
import pickle

patch_size = 8
patch_stride = 1

spsa_gamma = 1. / 6
spsa_alpha = lambda x: 0.25 / (x ** spsa_gamma)
spsa_beta = lambda x: 15. / (x ** (spsa_gamma / 4))
n_filters = 500

data_generator = svhn_data.SVHNData('/home/a.boiarov/Projects/datasets/SVHN', patch_size, patch_stride)
data_generator.get_mean_std(100)

# data_generator.save_example()

clustering = spsa_clustering.ClusteringSPSA(n_clusters=n_filters, data_shape=patch_size * patch_size, Gammas=None,
                                            alpha=spsa_alpha, beta=spsa_beta, norm_init=False, eta=900)

train_generator = data_generator.generate('train')
# spsa_train_num = data_generator.train_number
spsa_train_num = 1000

num = 0
for _ in range(spsa_train_num):
    print(num)
    num += 1
    train_data = next(train_generator)
    for patch in train_data[0]:
        patch = patch.flatten()
        clustering.fit(patch)

centers_fname = '/home/a.boiarov/Projects/spsa_clustering_gmm_log/centers.npy'
np.save(centers_fname, clustering.cluster_centers_)

features_gen = fg.FeaturesGenerator(clustering.cluster_centers_, patch_size, patch_stride, 5, 5, data_generator.mean,
                                    data_generator.std)

print('Generate features')

cl_train_num = data_generator.train_number
# cl_train_num = 2000
# cl_train_num = 100
X_train = []
y_train = []
num = 0
for i in range(cl_train_num-2):
    if i % 1000 == 0:
        print(i)
    train_data = next(train_generator)
    x = features_gen.forward(data_generator.train_gray[i])
    X_train.append(x)
    y_train.append(data_generator.train_labels[i])

print('Learn classifier')

clf = svm.LinearSVC()
clf.fit(X_train, y_train)

model_fname = '/home/a.boiarov/Projects/spsa_clustering_gmm_log/svm.sav'
pickle.dump(clf, open(model_fname, 'wb'))

