import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture, metrics
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
                                            beta=spsa_beta, norm_init=False, eta=700)

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

gmm = mixture.GaussianMixture(n_components=clust_means.shape[0], init_params='kmeans')
gmm.fit(data_set)
labels_pred_gmm = gmm.predict(data_set)

bgmm = mixture.BayesianGaussianMixture(n_components=clust_means.shape[0], init_params='random')
bgmm.fit(data_set)
labels_pred_bgmm = bgmm.predict(data_set)

ari_gmm = metrics.adjusted_rand_score(true_labels, labels_pred_gmm)
print('\nARI GMM: {:f}'.format(ari_gmm))
ari_bgmm = metrics.adjusted_rand_score(true_labels, labels_pred_bgmm)
print('ARI Bayesian GMM: {:f}'.format(ari_bgmm))
ari_spsa = metrics.adjusted_rand_score(true_labels, clustering.labels_)
print('ARI SPSA clustering: {:f}'.format(ari_spsa))

print('\n')
for i in range(clust_means.shape[0]):
    print('GMM covar matrix distance {0}: {1:f}'.format(i,
                                                        np.linalg.norm(clust_gammas[i] - gmm.covariances_[i])))
print('\n')
for i in range(clust_means.shape[0]):
    print('Bayesian GMM covar matrix distance {0}: {1:f}'.format(i,
                                                        np.linalg.norm(clust_gammas[i] - bgmm.covariances_[i])))

print('\n')
for i in range(clust_means.shape[0]):
    print('SPSA clustering covar matrix distance {0}: {1:f}'.format(i,
                                                        np.linalg.norm(clust_gammas[i] - clustering.Gammas[i])))

utils.plot_centers(clust_means, clustering)
utils.plot_centers_converg(clust_means, clustering)

utils.plot_clustering_cov(data_set, clustering.labels_, 'SPSA clustering partition', clustering.cluster_centers_,
                          clustering.Gammas)
utils.plot_clustering_cov(data_set, true_labels, 'True partition', clust_means, clust_gammas)
utils.plot_clustering_cov(data_set, labels_pred_gmm, 'GMM partition', gmm.means_, gmm.covariances_)
utils.plot_clustering_cov(data_set, labels_pred_bgmm, 'Bayesian GMM partition', bgmm.means_, bgmm.covariances_)
plt.show()


