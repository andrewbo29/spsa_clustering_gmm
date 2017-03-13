import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import utils
import kmeans_types
import spsa_clustering

import gmm_clustering


N = 5000
mix_prob = np.array([0.4, 0.4, 0.2])
clust_means = np.array([[0, 0], [2, 2], [-3, 6]])
clust_gammas = np.array([[[1, -0.7], [-0.7, 1]], np.eye(2), [[1, 0.8], [0.8, 1]]])
data_set = []
true_labels = []

noise_0 = spsa_clustering.Noise(func=lambda x: 0, name='0')
noise_1 = spsa_clustering.Noise(func=lambda x: np.random.normal(size=x.shape[0]), name='$\mathcal{N}(0,1)$')
noise_2 = spsa_clustering.Noise(func=lambda x: np.random.normal(0., 2., size=x.shape[0]),
                name='$\mathcal{N}(0,\sqrt{2})$')
noise_3 = spsa_clustering.Noise(func=lambda x: np.random.normal(1., 1., size=x.shape[0]),
                name='$\mathcal{N}(1,1)$')
noise_4 = spsa_clustering.Noise(func=lambda x: np.random.normal(1., 2., size=x.shape[0]),
                name='$\mathcal{N}(1,\sqrt{2})$')
noise_5 = spsa_clustering.Noise(func=lambda x: 10 * (np.random.rand(x.shape[0]) * 4 - 2),
                name='random')
noise_6 = spsa_clustering.Noise(func=lambda x: 0.1 * np.sin(np.arange(x.shape[0])+1) + 19 * np.sign(50 - (np.arange(x.shape[0])+1) % 100),
                name='irregular')
noise_7 = spsa_clustering.Noise(func=lambda x: [20]*x.shape[0], name='constant')

experiment_noise = noise_3

spsa_gamma = 1. / 6
spsa_alpha = lambda x: 0.25 / (x ** spsa_gamma)
spsa_beta = lambda x: 15. / (x ** (spsa_gamma / 4))

# spsa_alpha = lambda x: 0.001
# spsa_beta = lambda x: 0.001

clustering = spsa_clustering.ClusteringSPSA(n_clusters=3, data_shape=2, Gammas=None, alpha=spsa_alpha,
                                            beta=spsa_beta, norm_init=False, noise=experiment_noise,
                                            eta=None)

clustering_cov = spsa_clustering.ClusteringSPSA(n_clusters=3, data_shape=2, Gammas=None, alpha=spsa_alpha,
                                                beta=spsa_beta, norm_init=False, noise=experiment_noise,
                                                eta=1000)

for _ in range(N):
    mix_ind = np.random.choice(len(mix_prob), p=mix_prob)
    data_point = np.random.multivariate_normal(clust_means[mix_ind],
                                               clust_gammas[mix_ind])
    data_set.append(data_point)
    true_labels.append(mix_ind)
    clustering.fit(data_point)
    clustering_cov.fit(data_point)
data_set = np.array(data_set)

utils.order_clust_centers(clust_means, clustering)
clustering.clusters_fill(data_set)

clustering_cov.clusters_fill(data_set)

kmeans = kmeans_types.KMeansClassic(n_clusters=clust_means.shape[0], n_init=1, kmeans_pp=False, noise=experiment_noise)
kmeans.fit(data_set)

gmm = gmm_clustering.GMM(k=3, noise=experiment_noise)
gmm.fit_EM(data_set, max_iters=50)
gmm_predict = []
for data_point in data_set:
    gmm_predict.append(np.argmax(gmm.predict(data_point)))

ari_kmeans = metrics.adjusted_rand_score(true_labels, kmeans.labels_)
print('\nARI k-means: {:f}'.format(ari_kmeans))
ari_spsa = metrics.adjusted_rand_score(true_labels, clustering.labels_)
print('ARI SPSA clustering: {:f}'.format(ari_spsa))
ari_spsa_cov = metrics.adjusted_rand_score(true_labels, clustering_cov.labels_)
print('ARI SPSA covariance clustering: {:f}'.format(ari_spsa_cov))
ari_gmm = metrics.adjusted_rand_score(true_labels, gmm_predict)
print('ARI GMM: {:f}'.format(ari_gmm))

utils.plot_clustering(data_set, clustering.labels_, 'SPSA clustering partition with {0} noise'.format(clustering.noise))
utils.plot_clustering(data_set, true_labels, 'True partition')
utils.plot_clustering(data_set, kmeans.labels_, 'K-means partition with {0} noise'.format(clustering.noise))

utils.plot_clustering_cov(data_set, clustering_cov.labels_, 'SPSA covariance clustering partition with {0} noise'.
                          format(experiment_noise), clustering_cov.cluster_centers_, clustering_cov.Gammas)
utils.plot_clustering_cov(data_set, true_labels, 'True partition covariance', clust_means, clust_gammas)

plt.show()