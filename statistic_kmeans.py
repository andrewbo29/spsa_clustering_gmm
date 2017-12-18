import numpy as np
from sklearn import cluster, metrics
import utils
import spsa_clustering
import pam
from sklearn.metrics.pairwise import pairwise_distances


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
                                            beta=spsa_beta, norm_init=False, verbose=False)

kmeans = cluster.KMeans(n_clusters=clust_means.shape[0])
mb_kmeans = cluster.MiniBatchKMeans(n_clusters=clust_means.shape[0], n_init=1, init='random', max_iter=1, batch_size=1,
                                    max_no_improvement=None)

# dbscan = cluster.DBSCAN(n_jobs=-1)
# aff_prob = cluster.AffinityPropagation()

n_run = 100

ari_kmeans = np.zeros(n_run)
ari_mb_kmeans = np.zeros(n_run)
ari_spsa = np.zeros(n_run)
ari_pam = np.zeros(n_run)
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

    labels_pred_kmenas = kmeans.fit_predict(data_set)
    labels_pred_mb_kmeans = mb_kmeans.fit_predict(data_set)

    dist = pairwise_distances(data_set)
    labels_pred_pam = pam.cluster(dist, k=clust_means.shape[0])[0]

    # labels_pred_dbscan = dbscan.fit_predict(data_set)
    # labels_pred_aff_prob = aff_prob.fit_predict(data_set)

    ari_kmeans[i] = metrics.adjusted_rand_score(true_labels, labels_pred_kmenas)
    ari_mb_kmeans[i] = metrics.adjusted_rand_score(true_labels, labels_pred_mb_kmeans)
    ari_spsa[i] = metrics.adjusted_rand_score(true_labels, clustering.labels_)
    ari_pam[i] = metrics.adjusted_rand_score(true_labels, labels_pred_pam)
    # ari_dbscan[i] = metrics.adjusted_rand_score(true_labels, labels_pred_dbscan)
    # ari_aff_prop[i] = metrics.adjusted_rand_score(true_labels, labels_pred_aff_prob)

print('\nMean ARI k-means: {:f}'.format(ari_kmeans.mean()))
print('Mean ARI online k-means: {:f}'.format(ari_mb_kmeans.mean()))
print('Mean ARI SPSA clustering: {:f}'.format(ari_spsa.mean()))
print('\nMean ARI PAM: {:f}'.format(ari_pam.mean()))
# print('\nMean ARI DBSCAN: {:f}'.format(ari_dbscan.mean()))
# print('\nMean ARI Aff Prop: {:f}'.format(ari_aff_prop.mean()))
