import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, metrics
import utils
import spsa_clustering


N = 5000
mix_prob = np.array([0.4, 0.4, 0.2])
clust_means = np.array([[0, 0], [2, 2], [-3, 6]])
clust_gammas = np.array([[[1, -0.7], [-0.7, 1]], np.eye(2), [[1, 0.8], [0.8, 1]]])
data_set = []
true_labels = []

spsa_gamma = 1. / 6
spsa_alpha = lambda x: 0.25 / (x ** spsa_gamma)
spsa_beta = lambda x: 15. / (x ** (spsa_gamma / 4))

# spsa_alpha = lambda x: 0.001
# spsa_beta = lambda x: 0.001

clustering = spsa_clustering.ClusteringSPSA(n_clusters=clust_means.shape[0], data_shape=2, Gammas=None, alpha=spsa_alpha,
                                            beta=spsa_beta, norm_init=False)

for _ in range(N):
    mix_ind = np.random.choice(len(mix_prob), p=mix_prob)
    data_point = np.random.multivariate_normal(clust_means[mix_ind],
                                               clust_gammas[mix_ind])
    data_set.append(data_point)
    true_labels.append(mix_ind)
    clustering.fit(data_point)
data_set = np.array(data_set)

utils.order_clust_centers(clust_means, clustering)
clustering.clusters_fill(data_set)

kmeans = cluster.KMeans(n_clusters=clust_means.shape[0], n_init=1, init='random', max_iter=1)
labels_pred_kmenas = kmeans.fit_predict(data_set)

mb_kmeans = cluster.MiniBatchKMeans(n_clusters=clust_means.shape[0], n_init=1, init='random', max_iter=1, batch_size=1,
                                    max_no_improvement=None)
labels_pred_mb_kmeans = mb_kmeans.fit_predict(data_set)

ari_kmeans = metrics.adjusted_rand_score(true_labels, labels_pred_kmenas)
print('\nARI k-means: {:f}'.format(ari_kmeans))
ari_mb_kmeans = metrics.adjusted_rand_score(true_labels, labels_pred_mb_kmeans)
print('ARI online k-means: {:f}'.format(ari_mb_kmeans))
ari_spsa = metrics.adjusted_rand_score(true_labels, clustering.labels_)
print('ARI SPSA clustering: {:f}'.format(ari_spsa))

plt.style.use('grayscale')

utils.plot_centers(clust_means, clustering)
utils.plot_centers_converg(clust_means, clustering)

# utils.plot_clustering(data_set, clustering.labels_, 'SPSA clustering partition')
# utils.plot_clustering(data_set, true_labels, 'True partition')
# utils.plot_clustering(data_set, labels_pred_kmenas, 'K-means partition')
# utils.plot_clustering(data_set, labels_pred_mb_kmeans, 'Online k-means partition')

plt.show()




