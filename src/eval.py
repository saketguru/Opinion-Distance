from utils import read_distance_matrix_from_file
from sklearn.metrics import silhouette_score
from clustering import spectral_clustering, kmeans
from sklearn import metrics


def get_evaluation_scores(X, ground_truth_labels, clusters, filepath, ignored_indices=(), result_file=None,
                          files=None):
    if result_file is not None:
        X = read_distance_matrix_from_file(result_file, (), len(ground_truth_labels))

    n_clusters_, predicted_labels = kmeans(X, clusters)
    ari_k = metrics.adjusted_rand_score(ground_truth_labels, predicted_labels)

    n_clusters_, predicted_labels = spectral_clustering(X, clusters)
    ari_s = metrics.adjusted_rand_score(ground_truth_labels, predicted_labels)

    filepath.write("Silhouttee, K-mean ARI, Spectral-ARI\n")
    filepath.write("%s,%s,%s\n" % (silhouette_score(X, ground_truth_labels), ari_k, ari_s))
