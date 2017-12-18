import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture, metrics, cluster
import utils
import spsa_clustering


N = 5000
mix_prob = np.array([0.4, 0.4, 0.2])
clust_means = np.array([[0, 0], [2, 2], [-3, 6]])
clust_gammas = np.array([[[1, -0.7], [-0.7, 1]], np.eye(2), [[1, 0.8], [0.8, 1]]])

spsa_gamma = 1. / 6
spsa_alpha = lambda x: 0.25 / (x ** spsa_gamma)
spsa_beta = lambda x: 15. / (x ** (spsa_gamma / 4))

# spsa_alpha = lambda x: 0.001
# spsa_beta = lambda x: 0.001

clustering = spsa_clustering.ClusteringSPSA(n_clusters=clust_means.shape[0], data_shape=2, Gammas=None, alpha=spsa_alpha,
                                            beta=spsa_beta, norm_init=False, eta=1000, verbose=False)

gmm = mixture.GaussianMixture(n_components=clust_means.shape[0], init_params='kmeans')
bgmm = mixture.BayesianGaussianMixture(n_components=clust_means.shape[0], init_params='random')
# dbscan = cluster.DBSCAN(n_jobs=-1)
# aff_prob = cluster.AffinityPropagation()

n_run = 100

ari_gmm = np.zeros(n_run)
ari_bgmm = np.zeros(n_run)
ari_spsa = np.zeros(n_run)
ari_dbscan = np.zeros(n_run)
ari_aff_prop = np.zeros(n_run)

for i in range(n_run):
    print('Run {0}'.format(i))

    data_set = []
    true_labels = []
    for _ in range(N):
        mix_ind = np.random.choice(len(mix_prob), p=mix_prob)
        data_point = np.random.multivariate_normal(clust_means[mix_ind], clust_gammas[mix_ind])
        data_set.append(data_point)
        true_labels.append(mix_ind)
        clustering.fit(data_point)
    data_set = np.array(data_set)

    utils.order_clust_centers(clust_means, clustering)
    clustering.clusters_fill(data_set)

    gmm.fit(data_set)
    labels_pred_gmm = gmm.predict(data_set)

    bgmm.fit(data_set)
    labels_pred_bgmm = bgmm.predict(data_set)

    # labels_pred_dbscan = dbscan.fit_predict(data_set)
    # labels_pred_aff_prob = aff_prob.fit_predict(data_set)

    ari_gmm[i] = metrics.adjusted_rand_score(true_labels, labels_pred_gmm)
    ari_bgmm[i] = metrics.adjusted_rand_score(true_labels, labels_pred_bgmm)
    ari_spsa[i] = metrics.adjusted_rand_score(true_labels, clustering.labels_)
    # ari_dbscan[i] = metrics.adjusted_rand_score(true_labels, labels_pred_dbscan)
    # ari_aff_prop[i] = metrics.adjusted_rand_score(true_labels, labels_pred_aff_prob)

print('\nMean ARI GMM: {:f}'.format(ari_gmm.mean()))
print('Mean ARI Bayesian GMM: {:f}'.format(ari_bgmm.mean()))
print('Mean ARI SPSA clustering: {:f}'.format(ari_spsa.mean()))
# print('\nMean ARI DBSCAN: {:f}'.format(ari_dbscan.mean()))
# print('\nMean ARI Aff Prop: {:f}'.format(ari_aff_prop.mean()))