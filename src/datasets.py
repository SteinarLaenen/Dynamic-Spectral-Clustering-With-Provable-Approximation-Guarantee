from graph_tool import *
import graph_tool as gt
import graph_tool.generation
import graph_tool.spectral
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import fetch_openml
import numpy as np
import os
import joblib
import random
import time

from emnist import extract_training_samples, extract_test_samples


def load_emnist(subset):
    emnist_data_path = f'./emnist_data/{subset}.pkl'

    if os.path.exists(emnist_data_path):
        emnist = joblib.load(emnist_data_path)
    else:
        train_images, train_labels = extract_training_samples(subset)
        test_images, test_labels = extract_test_samples(subset)

        # Combine training and test sets
        X = np.concatenate((train_images, test_images))
        y = np.concatenate((train_labels, test_labels))

        # Flatten the images for compatibility
        X = X.reshape((X.shape[0], -1))

        emnist = {'data': X, 'target': y}
        os.makedirs('./emnist_data', exist_ok=True)
        joblib.dump(emnist, emnist_data_path)

    X = np.array(emnist['data'])
    y = np.array(emnist['target'])
    return X, y

def load_mnist():
    mnist_data_path = './mnist_data/mnist.pkl'
    if os.path.exists(mnist_data_path):
        mnist = joblib.load(mnist_data_path)
    else:
        mnist = fetch_openml('mnist_784', version=1)
        os.makedirs('./mnist_data', exist_ok=True)
        joblib.dump(mnist, mnist_data_path)
    X = np.array(mnist.data)
    y = np.array(mnist.target)

    return X,y


class KNN_Graph_Manager:
    def __init__(self, k=5, dataset_name="mnist", images_per_class=None):
        self.k = k
        self.dataset_name = dataset_name
        self.images_per_class = images_per_class
        self.graph_file_prefix = f'./{dataset_name}_data/{dataset_name}_knn_graph_k{self.k}_ipc{images_per_class if images_per_class else "all"}'
        self.graph = None
        self.labels = None

    def _get_graph_filename(self):
        return f"{self.graph_file_prefix}.gt"

    def build_or_load_graph(self):
        graph_filename = self._get_graph_filename()

        if os.path.exists(graph_filename):
            print(f"Loading graph from {graph_filename}")
            self.graph = load_graph(graph_filename)
            self.labels = self.graph.vertex_properties["labels"]
        else:
            print("Building the graph...")
            X, y = self._load_data()
            self._construct_knn_graph(X, y)
            os.makedirs(os.path.dirname(graph_filename), exist_ok=True)
            self.graph.save(graph_filename)
            print(f"Graph saved to {graph_filename}")

    def _load_data(self):
        if self.dataset_name == 'mnist':
            X, y = load_mnist()
        elif self.dataset_name == 'emnist':
            X, y = load_emnist('letters')
        else:
            raise ValueError("Unsupported dataset")

        if self.images_per_class is not None:
            X, y = self._select_subset(X, y)

        return X, y

    def _select_subset(self, X, y):
        unique_labels = np.unique(y)
        selected_indices = []

        for label in unique_labels:
            indices = np.where(y == label)[0]
            if len(indices) > self.images_per_class:
                selected_indices.extend(np.random.choice(indices, self.images_per_class, replace=False))
            else:
                selected_indices.extend(indices)

        return X[selected_indices], y[selected_indices]

    def _construct_knn_graph(self, X, y):
        self.graph = Graph(directed=False)
        self.labels = self.graph.new_vertex_property("string")

        # Add vertices and set labels
        for label in y:
            v = self.graph.add_vertex()
            self.labels[v] = label


        self.graph.vertex_properties["labels"] = self.labels

        # Construct KNN
        nn = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        distances, indices = nn.kneighbors(X)

        # Add edges
        for i in range(X.shape[0]):
            for j in indices[i][1:]:  # skip self-loop
                self.graph.add_edge(i, j)

        gt.generation.remove_parallel_edges(self.graph)

    def get_vertices_by_labels(self, target_labels):
        """
        Given a list of target labels, return all vertices belonging to those labels.

        :param target_labels: List of target labels.
        :return: List of vertices belonging to the target labels.
        """
        target_vertices = [v for v in self.graph.vertices() if self.graph.vertex_properties["labels"][v] in target_labels]
        return target_vertices

    def get_edges_between_sets(self, A, B):
        """
        Efficiently returns edges from set B to A and edges within B using Graph-tool's capabilities
        and numpy for advanced indexing. If set A is empty, it returns edges within B only.

        :param A: List or container of vertex indices in set A.
        :param B: List or container of vertex indices in set B.
        :return: List of edges from set B to A and within B.
        """
        # Convert vertex lists to numpy arrays for advanced indexing
        A_indices = np.array(A, dtype=np.int32)
        B_indices = np.array(B, dtype=np.int32)

        # Get the source and target vertex indices for all edges as NumPy arrays
        edge_sources_indices = np.array(self.graph.get_edges()[:, 0], dtype=np.int32)
        edge_targets_indices = np.array(self.graph.get_edges()[:, 1], dtype=np.int32)

        # Create masks for filtering edges
        # Mask for edges where both source and target are in B
        mask_in_B = np.in1d(edge_sources_indices, B_indices) & np.in1d(edge_targets_indices, B_indices)

        # Mask for edges from B to A (only if A is not empty)
        mask_from_B_to_A = np.array([False] * len(self.graph.get_edges()))  # Default to False for all edges
        if A:  # Only calculate if A is not empty
            mask_from_B_to_A = np.in1d(edge_sources_indices, B_indices) & np.in1d(edge_targets_indices, A_indices)

        # Combine masks
        final_mask = mask_in_B | mask_from_B_to_A

        # Select edges based on the combined mask
        selected_edges = self.graph.get_edges()[final_mask]
        # selected_edges = [self.graph.edge(i) for i, included in enumerate(final_mask) if included]

        return selected_edges

def build_initial_graph(knn_manager, classes):
    """Builds the initial graph for the given classes."""
    initial_graph = Graph(directed=False)
    vertices = knn_manager.get_vertices_by_labels(classes)

    edges = knn_manager.get_edges_between_sets([], vertices)

    original_indices = initial_graph.add_edge_list(edges, hashed=True)

    # store the old vertex to new vertex mapping
    initial_graph.vp["original_indices"] = original_indices

    return initial_graph, vertices
