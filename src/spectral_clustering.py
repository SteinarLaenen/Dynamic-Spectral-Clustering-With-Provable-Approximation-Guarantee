from graph_tool import Graph
import graph_tool as gt
import numpy as np
import time


from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

from sklearn.cluster import KMeans
import scipy
import scipy.sparse as sp

def spectral_clustering(graph: Graph, num_clusters: int, num_eigenvectors=None, norm_vectors=False) -> list[list[int]]:
    """
    Perform spectral clustering on the given graph object.

    :param graph: an Graph object
    :param num_clusters: the number of clusters to find
    :param num_eigenvectors: (optional) the number of eigenvectors to use to find the clusters
    :return: A list of lists. Each list corresponds to the indices of the vertex in that cluster.

    :raises ValueError: if the requested number of clusters or eigenvectors are not a positive integer
    """
    # If the number of eigenvectors is not specified, use the same number as the number of clusters we are looking for.
    if num_eigenvectors is None:
        num_eigenvectors = num_clusters

    # If the number of eigenvectors, or the number of clusters is 0, we should raise an error
    if num_eigenvectors <= 0:
        raise ValueError("You must use more than 0 eigenvectors for spectral clustering.")
    if num_clusters <= 0:
        raise ValueError("You must find at least 1 cluster when using spectral clustering.")
    if not isinstance(num_clusters, int) or not isinstance(num_eigenvectors, int):
        raise TypeError("The number of clusters and eigenvectors must be positive integers.")

    # Get the normalised laplacian matrix of the graph
    if 'edge_weight' in graph.ep.keys():
        adjacency_matrix = gt.spectral.adjacency(graph, weight=graph.ep['edge_weight'])
        laplacian_matrix = compute_normalized_laplacian(adjacency_matrix)
    else:
        laplacian_matrix = gt.spectral.laplacian(graph, norm=True, weight=None)
    # Find the bottom eigenvectors of the laplacian matrix
    _, eigenvectors = scipy.sparse.linalg.eigsh(laplacian_matrix, num_eigenvectors, which='SM')

    if norm_vectors:
         # Calculate the degree of each vertex. For undirected graphs, use 'total'; for directed, use 'in' or 'out'
        degrees = graph.get_total_degrees(graph.get_vertices(), eweight=graph.ep['edge_weight']) if 'edge_weight' in graph.ep.keys() else graph.get_total_degrees(graph.get_vertices())
        # Convert degrees to a numpy array and compute the square root
        sqrt_degrees = np.sqrt(degrees)
        # Normalize each eigenvector by the sqrt(d_u) for every vertex u
        # Ensure not to divide by zero by replacing zeros in sqrt_degrees with a very small number
        sqrt_degrees[sqrt_degrees == 0] = np.finfo(float).eps
        eigenvectors = eigenvectors / sqrt_degrees[:, np.newaxis]

    # Perform k-means on the eigenvectors to find the clusters
    labels = KMeans(n_clusters=num_clusters, n_init=10).fit_predict(eigenvectors)
    # Split the clusters.
    clusters = [[] for _ in range(num_clusters)]
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    return clusters

def compute_normalized_laplacian(adjacency_csr):
    """
    Compute the normalized Laplacian matrix from a given CSR adjacency matrix.

    Parameters:
    - adjacency_csr: scipy.sparse.csr_matrix, the adjacency matrix in CSR format

    Returns:
    - L_sym: scipy.sparse.csr_matrix, the normalized Laplacian matrix in CSR format
    """
    # Number of vertices
    n_vertices = adjacency_csr.shape[0]

    # Compute the degree of each vertex (including self loops)
    degrees = adjacency_csr.sum(axis=1).A1

    # Avoid division by zero for isolated vertices by setting their degree to 1 (will not affect the result)
    degrees[degrees == 0] = 1

    # Compute D^(-1/2)
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))

    # Compute the normalized Laplacian: L_sym = I - D^(-1/2) * A * D^(-1/2)
    I = sp.eye(n_vertices, format='csr')
    L_sym = I - D_inv_sqrt.dot(adjacency_csr).dot(D_inv_sqrt)

    return L_sym

def clusters_to_labels(clusters, num_data_points=None):
    """
    Given a list of clusters (a list of lists), convert it to a list of labels.

    :param clusters: A list of lists giving the members of each cluster
    :param num_data_points: The total number of data points in the data set
    :return: A single list containing the label for each datapoint.
    """
    if num_data_points is None:
        num_data_points = sum([len(cluster) for cluster in clusters])

    labels = [-1] * num_data_points

    for i, cluster in enumerate(clusters):
        for elem in cluster:
            try:
                labels[elem] = i
            except:
                pass



    return labels

def run_spectral_clustering(graph, true_labels, n_clusters):
    start_time = time.time()

    #run spectral clustering
    clusters = spectral_clustering(graph, n_clusters)

    labels = clusters_to_labels(clusters)

    end_time = time.time()
    runtime = end_time - start_time

    # Compute Adjusted Rand Index
    ari = adjusted_rand_score(true_labels, labels)



    return clusters, labels, ari, runtime
