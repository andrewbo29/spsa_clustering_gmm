import numpy as np
from scipy.spatial.distance import mahalanobis


class Noise(object):
    """Noise object"""

    def __init__(self, func=None, name=None):
        self.func = func
        self.name = name

    def fabric(self, x):
        return self.func(x)

    def __repr__(self):
        return self.name


class ClusteringSPSA(object):
    """Gaussian mixture model SPSA clustering"""

    def __init__(self, n_clusters, data_shape, Gammas=None, alpha=lambda x: 0.001, beta=lambda x: 0.001, verbose=True,
                 norm_init=False, noise=Noise(func=lambda x: 0, name='0'), eta=None):
        self.n_clusters = n_clusters
        self.Gammas = Gammas
        self.labels_ = np.zeros(0)
        self.cluster_centers_ = []
        self.alpha = alpha
        self.beta = beta
        self.norm_init = norm_init
        self.noise = noise
        self.eta = eta
        self.cluster_centers_list = []
        self.iteration_num = 1
        self.verbose = verbose

        if self.Gammas is None:
            self.Gammas = [np.eye(data_shape) for _ in range(self.n_clusters)]
        if self.eta is None:
            self.Gammas_inv = [np.linalg.inv(self.Gammas[i]) for i in range(self.n_clusters)]

    def fit(self, w):
        if self.verbose and (self.iteration_num % 100 == 0 or self.iteration_num == 1):
            print('SPSA clustering iteration: {0}'.format(self.iteration_num))

        if self.norm_init:
            if self.iteration_num == 1:
                self.cluster_centers_ = np.random.multivariate_normal(np.zeros(w.shape[0]), np.eye(w.shape[0]),
                                                                      size=self.n_clusters)
                self.cluster_centers_list.append(self.cluster_centers_.copy())
            self.fit_step(w)
        else:
            if self.iteration_num <= self.n_clusters:
                self.cluster_centers_.append(w)
            else:
                if self.iteration_num == self.n_clusters + 1:
                    self.cluster_centers_ = np.array(self.cluster_centers_)
                    self.cluster_centers_list.append(self.cluster_centers_.copy())
                self.fit_step(w)

        self.iteration_num += 1

    def y_vec(self, centers, w):
        if self.eta is None:
            return np.array([mahalanobis(w, centers[label], self.Gammas_inv[label]) + self.noise.fabric(
                self.iteration_num) for label in range(self.n_clusters)])
        else:
            return np.array([mahalanobis(w, centers[label], np.linalg.inv(self.Gammas[label])) + self.noise.fabric(
                    self.iteration_num) for label in range(self.n_clusters)])

    def j_vec(self, w):
        vec = np.zeros(self.n_clusters)
        vec[np.argmin(self.y_vec(self.cluster_centers_, w))] = 1
        return vec

    def delta_fabric(self, d):
        return np.where(np.random.binomial(1, 0.5, size=d) == 0, -1, 1)

    def alpha_fabric(self):
        return self.alpha(self.iteration_num)

    def beta_fabric(self):
        return self.beta(self.iteration_num)

    def fit_step(self, w):
        delta_n_t = self.delta_fabric(w.shape[0])[np.newaxis]
        alpha_n = self.alpha_fabric()
        beta_n = self.beta_fabric()

        j_vec = self.j_vec(w)[np.newaxis].T
        j_vec_dot_delta_t = np.dot(j_vec, delta_n_t)

        y_plus = self.y_vec(self.cluster_centers_ + beta_n * j_vec_dot_delta_t, w)[np.newaxis]

        y_minus = self.y_vec(self.cluster_centers_ - beta_n * j_vec_dot_delta_t, w)[np.newaxis]

        if self.eta is not None:
            cluster_ind = np.argmax(j_vec == 1)
            sub_mat = (self.cluster_centers_[cluster_ind] - w)[np.newaxis]
            scatter_matrix = np.dot(sub_mat.T, sub_mat)
            scale_param = np.tanh(self.iteration_num / self.eta) if self.iteration_num > self.eta else 0
            self.Gammas[cluster_ind] += scale_param * (scatter_matrix - self.Gammas[cluster_ind]) / self.iteration_num

        self.cluster_centers_ -= j_vec_dot_delta_t * np.dot(alpha_n * (y_plus - y_minus) / (2. * beta_n), j_vec)

        self.cluster_centers_list.append(self.cluster_centers_.copy())

    def cluster_decision(self, point):
        return np.argmin(
                [mahalanobis(point, self.cluster_centers_[label], self.Gammas_inv[label])
                 for label in range(self.n_clusters)])

    def clusters_fill(self, data):
        self.Gammas_inv = [np.linalg.inv(self.Gammas[label]) for label in range(self.n_clusters)]
        self.labels_ = np.zeros(data.shape[0])
        for ind, point in enumerate(data):
            self.labels_[ind] = self.cluster_decision(point)

    def centers_improve(self, data):
        for label in range(self.n_clusters):
            self.cluster_centers_[label] = (self.cluster_centers_[label] + np.mean(data[self.labels_ == label], axis=0)) / 2

