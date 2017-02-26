import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, cluster
import pandas as pd
import utils
import spsa_clustering

# data_filename = '../data/mnist_1000.csv'
data_filename = '../data/mnist_train.csv'

df = pd.read_csv(data_filename)
df /= 255
mean_vec = np.array(df.mean()[1:].tolist())

noise_0 = spsa_clustering.Noise(func=lambda x: 0, name='0')
noise_1 = spsa_clustering.Noise(func=lambda x: np.random.normal(), name='$\mathcal{N}(0,1)$')
noise_2 = spsa_clustering.Noise(func=lambda x: np.random.normal(0., 2.),
                name='$\mathcal{N}(0,\sqrt{2})$')
noise_3 = spsa_clustering.Noise(func=lambda x: np.random.normal(1., 1.),
                name='$\mathcal{N}(1,1)$')
noise_4 = spsa_clustering.Noise(func=lambda x: np.random.normal(1., 2.),
                name='$\mathcal{N}(1,\sqrt{2})$')
noise_5 = spsa_clustering.Noise(func=lambda x: 10 * (np.random.rand() * 4 - 2),
                name='random')
noise_6 = spsa_clustering.Noise(func=lambda x: 0.1 * np.sin(x) + 19 * np.sign(50 - x % 100),
                name='irregular')
noise_7 = spsa_clustering.Noise(func=lambda x: 20, name='constant')

experiment_noise = noise_0

spsa_gamma = 1. / 6
spsa_alpha = lambda x: 0.25 / (x ** spsa_gamma)
spsa_beta = lambda x: 15. / (x ** (spsa_gamma / 4))

# spsa_alpha = lambda x: 0.0001
# spsa_beta = lambda x: 0.0001

clustering = spsa_clustering.ClusteringSPSA(n_clusters=10, data_shape=784, Gammas=None, alpha=spsa_alpha,
                                            beta=spsa_beta, norm_init=False, noise=experiment_noise)

data_set = []
true_labels = []

# init_ind = []
# for label in range(10):
#     ind = np.random.choice(df.index[df['label'] == label].tolist(), 1)
#     row = df.loc[ind[0], :]
#     true_labels.append(row[0])
#     data_point = np.array(row[1:].tolist(), dtype=float)
#     data_set.append(data_point)
#     clustering.fit(data_point)
#     init_ind.append(ind)

index = list(range(df.shape[0]))
np.random.shuffle(index)
for i in index:
    # if i not in init_ind:
    row = df.loc[i, :]
    true_labels.append(row[0])
    data_point = np.array(row[1:].tolist(), dtype=float) - mean_vec
    data_set.append(data_point)
    clustering.fit(data_point)
data_set = np.array(data_set)

# clustering.clusters_fill(data_set)
# clustering.centers_improve(data_set)

# kmeans = cluster.KMeans(n_clusters=10, n_init=1, init='random', max_iter=1)
# labels_pred_kmenas = kmeans.fit_predict(data_set)
#
# ari_kmeans = metrics.adjusted_rand_score(true_labels, labels_pred_kmenas)
# print('\nARI k-means: {:f}'.format(ari_kmeans))
# ari_spsa = metrics.adjusted_rand_score(true_labels, clustering.labels_)
# print('ARI SPSA clustering: {:f}'.format(ari_spsa))

utils.plot_mnist_centers(clustering)
plt.show()




