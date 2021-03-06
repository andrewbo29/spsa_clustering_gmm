# coding=utf-8
import sys, os, pickle
from datetime import datetime

from scipy.stats import cauchy, halfcauchy
import numpy as np
from sklearn import metrics, cluster
import matplotlib.pyplot as plt
from tqdm import tqdm
import pam
from sklearn.metrics.pairwise import pairwise_distances

import spsa_clustering
import utils


def get_sparse_gmm_model(clust_num, data_shape):
    sigma_mu = utils.positive_distr(cauchy.rvs, data_shape)
    mu_list = [np.random.normal(0, sigma_mu, size=data_shape) for _ in range(clust_num)]
    sigma_list = utils.positive_distr(halfcauchy.rvs, clust_num)

    alpha_g = 10
    e_zero = np.random.gamma(alpha_g, clust_num * alpha_g)
    w_list = np.random.dirichlet([e_zero] * clust_num)

    return mu_list, sigma_list, w_list


def get_sparse_gmm_example(data_shape):
    w_list = [0.4, 0.6]
    mu_list = [np.random.randint(7, size=data_shape) for _ in range(2)]
    sigma_list = [1, 1]

    return mu_list, sigma_list, w_list


def stat():
    clust_num = 3
    data_shape = 2

    mu_list, sigma_list, w_list = get_sparse_gmm_model(clust_num, data_shape)

    spsa_gamma = 1. / 6
    spsa_alpha = lambda x: 0.25 / (x ** spsa_gamma)
    spsa_beta = lambda x: 15. / (x ** (spsa_gamma / 4))

    # spsa_alpha = lambda x: 0.001
    # spsa_beta = lambda x: 0.001

    n_run = 10
    N = 3000

    ari_spsa = np.zeros(n_run)
    ari_kmeans = np.zeros(n_run)
    ari_mb_kmeans = np.zeros(n_run)
    ari_pam = np.zeros(n_run)

    cent_dist = np.zeros(n_run)
    cent_dist_kmeans = np.zeros(n_run)
    cent_dist_mb_kmeans = np.zeros(n_run)
    cent_dist_pam = np.zeros(n_run)

    for i in tqdm(range(n_run)):
        clustering = spsa_clustering.ClusteringSPSA(n_clusters=clust_num, data_shape=data_shape, Gammas=None,
                                                    alpha=spsa_alpha,
                                                    beta=spsa_beta, norm_init=False, verbose=False, sparse=True, eta=700,
                                                    spsa_sigma=False)

        kmeans = cluster.KMeans(n_clusters=clust_num)
        mb_kmeans = cluster.MiniBatchKMeans(n_clusters=clust_num, n_init=1, init='random', max_iter=1,
                                            batch_size=1,
                                            max_no_improvement=None)

        data_set = []
        true_labels = []
        for _ in range(N):
            mix_ind = np.random.choice(len(w_list), p=w_list)
            data_point = np.random.multivariate_normal(mu_list[mix_ind], np.identity(data_shape) * sigma_list[mix_ind])
            data_set.append(data_point)
            true_labels.append(mix_ind)
            # clustering.fit(data_point)
        data_set = np.array(data_set)

        # utils.order_clust_centers(np.array(mu_list), clustering)

        # clustering.clusters_fill(data_set)

        labels_pred_kmenas = kmeans.fit_predict(data_set)
        labels_pred_mb_kmeans = mb_kmeans.fit_predict(data_set)

        dist = pairwise_distances(data_set)
        labels_pred_pam, pam_med = pam.cluster(dist, k=clust_num)

        # ari_spsa[i] = metrics.adjusted_rand_score(true_labels, clustering.labels_)
        # cent_dist[i] = utils.mean_cent_dist(np.array(mu_list), clustering)

        ari_kmeans[i] = metrics.adjusted_rand_score(true_labels, labels_pred_kmenas)
        ari_mb_kmeans[i] = metrics.adjusted_rand_score(true_labels, labels_pred_mb_kmeans)
        ari_pam[i] = metrics.adjusted_rand_score(true_labels, labels_pred_pam)

        cent_dist_kmeans[i] = utils.mean_cent_dist_(np.array(mu_list), kmeans.cluster_centers_)
        cent_dist_mb_kmeans[i] = utils.mean_cent_dist_(np.array(mu_list), mb_kmeans.cluster_centers_)
        cent_dist_pam[i] = utils.mean_cent_dist_(np.array(mu_list), data_set[pam_med])

    print(ari_spsa.mean(), cent_dist.mean())

    print('\nMean ARI k-means: {:f}, Mean L2: {:f}'.format(ari_kmeans.mean(), cent_dist_kmeans.mean()))
    print('Mean ARI online k-means: {:f}, Mean L2: {:f}'.format(ari_mb_kmeans.mean(), cent_dist_mb_kmeans.mean()))
    # print('Mean ARI SPSA clustering: {:f}, Mean L2: {:f}'.format(ari_spsa.mean(), cen))
    print('\nMean ARI PAM: {:f}, Mean L2: {:f}'.format(ari_pam.mean(), cent_dist_pam.mean()))
    # print('\nMean ARI DBSCAN: {:f}'.format(ari_dbscan.mean()))


def main():
    clust_num = 3
    data_shape = 2

    mu_list, sigma_list, w_list = get_sparse_gmm_model(clust_num, data_shape)

    spsa_gamma = 1. / 6
    spsa_alpha = lambda x: 0.25 / (x ** spsa_gamma)
    spsa_beta = lambda x: 15. / (x ** (spsa_gamma / 4))

    # spsa_alpha = lambda x: 0.001
    # spsa_beta = lambda x: 0.001

    clustering = spsa_clustering.ClusteringSPSA(n_clusters=clust_num, data_shape=data_shape, Gammas=None, alpha=spsa_alpha,
                                beta=spsa_beta, norm_init=False, verbose=False, sparse=False, eta=None)

    N = 3000
    data_set = []
    true_labels = []
    for _ in range(N):
        mix_ind = np.random.choice(len(w_list), p=w_list)
        data_point = np.random.multivariate_normal(mu_list[mix_ind], np.identity(data_shape) * sigma_list[mix_ind])
        data_set.append(data_point)
        true_labels.append(mix_ind)
        clustering.fit(data_point)
    data_set = np.array(data_set)

    dataset_name = 'good'
    dataset_dir = os.path.join('datasets', dataset_name)
    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)

    np.save(os.path.join(dataset_dir, 'data.npy'), data_set)
    np.save(os.path.join(dataset_dir, 'true.npy'), np.array(true_labels))

    param = {'mu': mu_list, 'sigma': sigma_list, 'w': w_list}
    with open(os.path.join(dataset_dir, 'param.pickle'), 'wb') as f:
        pickle.dump(param, f)

    utils.order_clust_centers(np.array(mu_list), clustering)

    clustering.clusters_fill(data_set)
    ari_spsa = metrics.adjusted_rand_score(true_labels, clustering.labels_)

    print('ARI: {}'.format(ari_spsa))
    print('Mean centers dist: {}'.format(utils.mean_cent_dist(np.array(mu_list), clustering)))

    utils.plot_centers(np.array(mu_list), clustering)
    # utils.plot_centers_converg(np.array(mu_list), clustering)

    utils.plot_clustering(data_set, clustering.labels_, 'SPSA clustering partition')
    utils.plot_clustering(data_set, true_labels, 'True partition')

    plt.show()


def load_experiment(name='bad'):
    dataset_dir = os.path.join('datasets', name)

    data_set = np.load(os.path.join(dataset_dir, 'data.npy'))
    true_labels = np.load(os.path.join(dataset_dir, 'true.npy'))

    with open(os.path.join(dataset_dir, 'param.pickle'), 'rb') as f:
        param = pickle.load(f)

    mu_list, sigma_list, w_list = param['mu'], param['sigma'], param['w']

    clust_num = len(mu_list)
    data_shape = data_set[0].shape[0]

    spsa_gamma = 1. / 6
    spsa_alpha = lambda x: 0.25 / (x ** spsa_gamma)
    spsa_beta = lambda x: 15. / (x ** (spsa_gamma / 4))

    clustering = spsa_clustering.ClusteringSPSA(n_clusters=clust_num, data_shape=data_shape, Gammas=None,
                                                alpha=spsa_alpha,
                                                beta=spsa_beta, norm_init=False, verbose=False, sparse=False, eta=None,
                                                spsa_sigma=False)

    rand_ind = np.random.permutation(data_set.shape[0])

    for i in rand_ind:
        clustering.fit(data_set[i])

    # utils.order_clust_centers(np.array(mu_list), clustering)

    clustering.clusters_fill(data_set[rand_ind])
    ari_spsa = metrics.adjusted_rand_score(true_labels[rand_ind], clustering.labels_)

    print('ARI: {}'.format(ari_spsa))
    print('Mean centers dist: {}'.format(utils.mean_cent_dist(np.array(mu_list), clustering)))

    utils.plot_centers(np.array(mu_list), clustering)
    # utils.plot_centers_converg(np.array(mu_list), clustering)

    # utils.plot_clustering(data_set[rand_ind], clustering.labels_, 'SPSA clustering partition')
    # utils.plot_clustering(data_set[rand_ind], true_labels[rand_ind], 'True partition')

    # for Gamma in clustering.Gammas:
    #     print(Gamma)

    # for center in clustering.cluster_centers_:
    #     print(center)

    # utils.plot_clustering_cov(data_set, clustering.labels_, 'SPSA clustering partition', clustering.cluster_centers_,
    #                           clustering.Gammas)

    plt.show()


if __name__ == '__main__':
    # sys.exit(main())
    # sys.exit(stat())
    # plt.style.use('grayscale')
    sys.exit(load_experiment('ugly'))
