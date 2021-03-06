import numpy as np
from sklearn import metrics
import utils
import kmeans_types
import spsa_clustering
import gmm_clustering
import pam
from sklearn.metrics.pairwise import pairwise_distances


N = 5000
mix_prob = np.array([0.4, 0.4, 0.2])
clust_means = np.array([[0, 0], [2, 2], [-3, 6]])
clust_gammas = np.array([[[1, -0.7], [-0.7, 1]], np.eye(2), [[1, 0.8], [0.8, 1]]])

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

noises = [noise_1, noise_2, noise_3, noise_4, noise_5, noise_6, noise_7]

noise_1_mul = spsa_clustering.Noise(func=lambda x: np.random.normal(size=x.shape[0]), name='$\mathcal{N}(0,1)$')
noise_2_mul = spsa_clustering.Noise(func=lambda x: np.random.normal(0., 2., size=x.shape[0]),
                name='$\mathcal{N}(0,\sqrt{2})$')
noise_3_mul = spsa_clustering.Noise(func=lambda x: np.random.normal(1., 1., size=x.shape[0]),
                name='$\mathcal{N}(1,1)$')
noise_4_mul = spsa_clustering.Noise(func=lambda x: np.random.normal(1., 2., size=x.shape[0]),
                name='$\mathcal{N}(1,\sqrt{2})$')
noise_5_mul = spsa_clustering.Noise(func=lambda x: 10 * (np.random.rand(x.shape[0]) * 4 - 2),
                name='random')
noise_6_mul = spsa_clustering.Noise(func=lambda x: 0.1 * np.sin(np.arange(x.shape[0])+1) + 19 * np.sign(50 - (np.arange(x.shape[0])+1) % 100),
                name='irregular')
noise_7_mul = spsa_clustering.Noise(func=lambda x: [20]*x.shape[0], name='constant')

noises_mul = [noise_1_mul, noise_2_mul, noise_3_mul, noise_4_mul, noise_5_mul, noise_6_mul, noise_7_mul]

spsa_gamma = 1. / 6
spsa_alpha = lambda x: 0.25 / (x ** spsa_gamma)
spsa_beta = lambda x: 15. / (x ** (spsa_gamma / 4))

# spsa_alpha = lambda x: 0.001
# spsa_beta = lambda x: 0.001

n_run = 10

for j in range(len(noises)):
    ari_kmeans = np.zeros(n_run)
    ari_spsa = np.zeros(n_run)
    ari_spsa_cov = np.zeros(n_run)
    ari_gmm = np.zeros(n_run)
    ari_pam = np.zeros(n_run)

    centers_dist_kmeans = np.zeros(n_run)
    centers_dist_spsa = np.zeros(n_run)
    centers_dist_spsa_cov = np.zeros(n_run)
    centers_dist_gmm = np.zeros(n_run)

    for i in range(n_run):
        print('Run {0}'.format(i))

        clustering = spsa_clustering.ClusteringSPSA(n_clusters=clust_means.shape[0], data_shape=2, Gammas=None,
                                                    alpha=spsa_alpha,
                                                    beta=spsa_beta, norm_init=False, verbose=False, noise=noises[j])

        clustering_cov = spsa_clustering.ClusteringSPSA(n_clusters=clust_means.shape[0], data_shape=2, Gammas=None,
                                                        alpha=spsa_alpha,
                                                        beta=spsa_beta, norm_init=False, noise=noises[j],
                                                        eta=3000, verbose=False)

        data_set = []
        true_labels = []
        for _ in range(N):
            mix_ind = np.random.choice(len(mix_prob), p=mix_prob)
            data_point = np.random.multivariate_normal(clust_means[mix_ind], clust_gammas[mix_ind])
            data_set.append(data_point)
            true_labels.append(mix_ind)
            # clustering.fit(data_point)
            # clustering_cov.fit(data_point)
        data_set = np.array(data_set)

        # utils.order_clust_centers(clust_means, clustering)
        # clustering.clusters_fill(data_set)
        # clustering_cov.clusters_fill(data_set)

        # kmeans = kmeans_types.KMeansClassic(n_clusters=clust_means.shape[0], n_init=1, kmeans_pp=False,
        #                                     noise=noises[j], verbose=False, max_iter=50)
        # kmeans.fit(data_set)

        # gmm = gmm_clustering.GMM(k=3, noise=noises_mul[j])
        # gmm.fit_EM(data_set, max_iters=100)
        # gmm_predict = []
        # for data_point in data_set:
        #     gmm_predict.append(np.argmax(gmm.predict(data_point)))

        dist = pairwise_distances(data_set)
        labels_pred_pam = pam.cluster(dist, k=clust_means.shape[0])[0]

        # ari_kmeans[i] = metrics.adjusted_rand_score(true_labels, kmeans.labels_)
        # ari_spsa[i] = metrics.adjusted_rand_score(true_labels, clustering.labels_)
        # ari_spsa_cov[i] = metrics.adjusted_rand_score(true_labels, clustering_cov.labels_)
        # ari_gmm[i] = metrics.adjusted_rand_score(true_labels, gmm_predict)
        ari_pam[i] = metrics.adjusted_rand_score(true_labels, labels_pred_pam)

        # centers_dist_kmeans[i] = utils.mean_sq_dist(clust_means, kmeans.cluster_centers_)
        # centers_dist_spsa[i] = utils.mean_sq_dist(clust_means, clustering.cluster_centers_)
        # centers_dist_spsa_cov[i] = utils.mean_sq_dist(clust_means, clustering_cov.cluster_centers_)
        # centers_dist_gmm[i] = utils.mean_sq_dist(clust_means, gmm.params.mu)

    # print('\nMean ARI k-means with noise {0}: {1:f}'.format(noises[j], ari_kmeans.mean()))
    # print('Mean ARI SPSA clustering with noise {0}: {1:f}'.format(noises[j], ari_spsa.mean()))
    # print('Mean ARI SPSA covariance clustering with noise {0}: {1:f}'.format(noises[j], ari_spsa_cov.mean()))
    # print('Mean ARI EM with noise {0}: {1:f}'.format(noises_mul[j], ari_gmm.mean()))
    print('Mean ARI PAM {0}: {1:f}'.format(noises_mul[j], ari_pam.mean()))

    # print('\nMean centers dist k-means with noise {0}: {1:f}'.format(noises[j], centers_dist_kmeans.mean()))
    # print('Mean centers dist SPSA clustering with noise {0}: {1:f}'.format(noises[j], centers_dist_spsa.mean()))
    # print('Mean centers dist SPSA covariance clustering with noise {0}: {1:f}'.format(noises[j], centers_dist_spsa_cov.mean()))
    # print('Mean centers dist EM with noise {0}: {1:f}'.format(noises_mul[j], centers_dist_gmm.mean()))