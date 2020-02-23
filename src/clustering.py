import numpy as np
import warnings
from sklearn.cluster import SpectralClustering, KMeans

warnings.filterwarnings("ignore", category=DeprecationWarning)


def spectral_clustering(X, clusters):
    spectral = SpectralClustering(n_clusters=clusters, eigen_solver='arpack', affinity="poly", assign_labels='kmeans',
                                  n_init=100)  #
    spectral.fit(X)
    labels = spectral.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    return n_clusters_, labels


def kmeans(X, clusters):
    kmeans = KMeans(init='k-means++', n_clusters=clusters, n_init=100)
    kmeans.fit(X)
    labels = kmeans.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    return n_clusters_, labels
