import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy import linalg
import os


COLORS = ['navy', 'darkgreen', 'darkred']


def plot_clustering(data, partition, title):
    df = pd.DataFrame(dict(x=data[:, 0], y=data[:, 1], g=partition))
    g = sns.lmplot('x', 'y', data=df, hue='g', markers=['o', '+', 'x'], fit_reg=False, legend=False)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(title, fontsize=12)


def plot_clustering_cov(data, partition, title, mean, covar):
    df = pd.DataFrame(dict(x=data[:, 0], y=data[:, 1], g=partition))
    g = sns.lmplot('x', 'y', data=df, hue='g', markers=['o', '+', 'x'], fit_reg=False, legend=False)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(title, fontsize=12)

    for i in range(mean.shape[0]):
        v, w = linalg.eigh(covar[i])
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean[i], v[0], v[1], 180. + angle, color=COLORS[i])

        splot = g.axes[0, 0]
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)


def plot_centers_converg(true_centers, clust):
    conv = np.sqrt(np.sum(((true_centers - np.array(clust.cluster_centers_list)) ** 2), axis=2))

    sns.set_style("darkgrid")
    plt.figure()
    plt.plot(conv[:, 0], '--', linewidth=3)
    plt.plot(conv[:, 1], '-.', linewidth=3)
    plt.plot(conv[:, 2], ':', linewidth=3)
    plt.legend(['Centroid %d' % (i + 1) for i in range(len(true_centers))])
    plt.xlabel('Iterations')
    plt.ylabel('Error norm')


def plot_centers(true_centers, clust):
    centers = np.array(clust.cluster_centers_list)

    col_num = 3
    if clust.n_clusters % col_num == 0:
        row_num = clust.n_clusters / col_num
    else:
        row_num = clust.n_clusters // col_num + 1

    sns.set_style("darkgrid")
    fig = plt.figure()
    for i in range(clust.n_clusters):
        ax = fig.add_subplot(row_num, col_num, i + 1)
        ax.plot(centers[:, i, 0], centers[:, i, 1], '-.', linewidth=3, zorder=1)
        ax.scatter(true_centers[i, 0], true_centers[i, 1], s=40, marker='s', c='r', zorder=2)
        ax.set_title('Centroid %d' % (i + 1))

    plt.subplots_adjust(hspace=0.3)


def order_clust_centers(true_centers, clustering):
    clust_num = true_centers.shape[0]
    order = np.zeros(clust_num, dtype=int)
    for i in range(clust_num):
        dist = np.zeros(clust_num)
        for j in range(clust_num):
            dist[j] = euclidean(true_centers[i], clustering.cluster_centers_[j])
        order[i] = np.argmin(dist)

    clustering.cluster_centers_ = clustering.cluster_centers_[order]

    for j in range(len(clustering.cluster_centers_list)):
        clustering.cluster_centers_list[j] = clustering.cluster_centers_list[j][order]


def make_mnist_subset(mnist_filename, per_category=1000):
    df = pd.read_csv(mnist_filename)

    frames = []
    for label in range(10):
        ind = np.random.choice(df.index[df['label'] == label].tolist(), per_category)
        frames.append(df.loc[ind, :])

    result = pd.concat(frames)
    result_filename = os.path.join(os.path.dirname(mnist_filename), 'mnist_{0}.csv'.format(per_category))
    result.to_csv(result_filename, index=False)


def plot_mnist_centers(clustering):
    sns.set_style("darkgrid")
    for i in range(clustering.n_clusters):
        plt.subplot(2, 5, i + 1)
        grid_data = clustering.cluster_centers_[i].reshape(28, 28)
        plt.imshow(grid_data, interpolation="none", cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()


def plot_mnist(df):
    sns.set_style("darkgrid")
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        ind = np.random.choice(df.index[df['label'] == i].tolist(), 1)
        row = df.loc[ind[0], :]
        data_point = np.array(row[1:].tolist(), dtype=float)
        grid_data = data_point.reshape(28, 28)
        plt.imshow(grid_data, interpolation="none", cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()


def mean_sq_dist(true_centers, pred_centers):
    return np.mean(np.sqrt(np.sum((true_centers - pred_centers) ** 2, axis=1)))
