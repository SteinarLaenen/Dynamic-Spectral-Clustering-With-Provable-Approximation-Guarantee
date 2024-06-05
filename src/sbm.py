from graph_tool import Graph
import graph_tool as gt
import graph_tool.generation
import numpy as np
import os
import random

import scipy
import scipy.sparse as sp

class DynamicSBMGraphMerge:
    def __init__(self, k, sizes, p, q):
        """
        Initializes a dynamic graph based on the stochastic block model (SBM) with clusters of varying sizes.

        Parameters:
        - k (int): Number of clusters.
        - sizes (list of int): Sizes of each cluster.
        - p (float): Probability of an edge within each cluster.
        - q (float): Probability of an edge between each pair of different clusters.
        """
        self.k = k
        self.sizes = sizes
        self.p = p
        self.q = q
        self.graph, self.labels = self._generate_sbm_graph()
        gt.generation.remove_self_loops(self.graph)
        self.available_vertices = set(range(sum(sizes)))

    def _generate_sbm_graph(self):
        """
        Generates a graph from a stochastic block model with given parameters.

        Returns:
        - Graph: A graph-tool Graph object.
        - labels (numpy.ndarray): Array of vertex cluster memberships.
        """
        import numpy as np
        import graph_tool.all as gt

        # Group membership for each node, adjusted for varying sizes
        b = np.concatenate([np.full(size, i) for i, size in enumerate(self.sizes)])

        # Initialize the edge propensity matrix
        probs = np.zeros((self.k, self.k))

        # Fill the diagonal with probabilities p adjusted for within-cluster edges
        for i in range(self.k):
            n_i = self.sizes[i]
            probs[i, i] = self.p * n_i * (n_i - 1) / 2  # Adjusted for the expected number of edges within a cluster

        # Fill off-diagonal with probabilities q adjusted for between-cluster edges
        for i in range(self.k):
            for j in range(i + 1, self.k):
                n_i, n_j = self.sizes[i], self.sizes[j]
                probs[i, j] = probs[j, i] = self.q * n_i * n_j  # Adjusted for the expected number of edges between clusters

        # Generate the SBM graph
        g = graph_tool.generation.generate_sbm(b, probs, directed=False)

        return g, b

    def overlay_erdos_renyi_optimized(self, s):
        """
        Efficiently overlays the graph with unique edges based on the Erdős–Rényi model parameter s.

        Parameters:
        - s (float): Probability of an edge existing between any two vertices.

        Returns:
        - new_edges (list): A list of new, unique edges added to the graph.
        """
        N = self.graph.num_vertices()
        # Calculate the expected number of new edges to add
        total_possible_edges = N * (N - 1) // 2
        num_edges_to_add = int(np.round(s * total_possible_edges))

        new_edges = set()
        attempts = 0
        max_attempts = num_edges_to_add * 10  # Set a limit to prevent infinite loops

        while len(new_edges) < num_edges_to_add and attempts < max_attempts:
            u, v = np.random.randint(0, N, size=2)
            if u != v and not self.graph.edge(u, v):
                # Add the edge if it's not a self-loop and doesn't already exist
                new_edges.add((min(u, v), max(u, v)))  # Ensure consistency in edge direction
            attempts += 1

        return list(new_edges)

    def merge_clusters(self, label_a, label_b, r):
        """
        Samples edges between two clusters with probability r without directly updating the graph.
        Returns the sampled edges as a 2xm numpy array and the updated labels array.

        Parameters:
        - label_a (int): The label of the first cluster.
        - label_b (int): The label of the second cluster.
        - r (float): The probability of adding an edge between any two vertices in the two clusters.

        Returns:
        - sampled_edges (numpy.ndarray): A 2xm array of sampled edges between the two clusters.
        - new_labels (numpy.ndarray): The updated labels after merging the two clusters.
        """
        import numpy as np

        # Find vertices belonging to each cluster
        vertices_a = np.where(self.labels == label_a)[0]
        vertices_b = np.where(self.labels == label_b)[0]

        # Initialize an empty list to store sampled edges
        sampled_edges_list = []

        # Sample edges between the two clusters with probability r
        for v_a in vertices_a:
            for v_b in vertices_b:
                if np.random.rand() < r:
                    sampled_edges_list.append([v_a, v_b])

        # Convert the list of sampled edges to a numpy array
        sampled_edges = sampled_edges_list  # Transpose to get a 2xm array

        # Update labels to reflect the merged cluster
        self.labels[vertices_b] = label_a

        return sampled_edges, self.labels



    def get_graph(self):
        """
        Returns the current state of the graph.

        Returns:
        - Graph: The current graph-tool Graph object.
        """
        return self.graph

    def get_labels(self):
        """
        Returns the current cluster memberships of all vertices.

        Returns:
        - labels (numpy.ndarray): The array of vertex cluster memberships.
        """
        return self.labels




class DynamicSBMGraph:
    def __init__(self, k, n, p, q):
        """
        Initializes a dynamic graph based on the stochastic block model (SBM).

        Parameters:
        - k (int): Number of clusters.
        - n (int): Size of each cluster.
        - p (float): Average number of expected edges within each cluster.
        - q (float): Average number of expected edges between each pair of different clusters.
        """
        self.k = k
        self.n = n
        self.p = p
        self.q = q
        self.graph, self.labels = self._generate_sbm_graph()
        gt.generation.remove_self_loops(self.graph)
        self.available_vertices = set(range(k * n))

    def _generate_sbm_graph(self):
        """
        Generates a graph from a stochastic block model with given parameters.

        Returns:
        - Graph: A graph-tool Graph object.
        - labels (numpy.ndarray): Array of vertex cluster memberships.
        """
        # Group membership for each node
        b = np.repeat(np.arange(self.k), self.n)

        # Edge propensity matrix calculation
        probs = np.full((self.k, self.k), self.q)
        np.fill_diagonal(probs, self.p)

        # Adjust p and q for expected number of edges rather than probabilities
        # For undirected graphs, the expecte
        self.available_vertices = set(range(k * n))



    def _generate_sbm_graph(self):
        """
        Generates a graph from a stochastic block model with given parameters.

        Returns:
        - Graph: A graph-tool Graph object.
        - labels (numpy.ndarray): Array of vertex cluster memberships.
        """
        # Group membership for each node
        b = np.repeat(np.arange(self.k), self.n)

        # Edge propensity matrix calculation
        probs = np.full((self.k, self.k), self.q)
        np.fill_diagonal(probs, self.p)

        # Adjust p and q for expected number of edges rather than probabilities
        # For undirected graphs, the expected number of edges within a cluster
        n = self.n
        probs = probs * (n * (n - 1) / 2) if self.k > 1 else probs * (n * (n))/2
        probs = probs * 2

        # Generate the SBM graph
        g = gt.generation.generate_sbm(b, probs, directed=False)

        return g, b

    def sample_internal_edges(self, n_small, r):
        """
        Randomly selects a subset of vertices, samples edges within it based on
        a given probability, and updates their cluster memberships.

        Parameters:
        - n_small (int): The size of the subset to select and internally connect.
        - r (float): The probability of adding an edge between any two vertices in the subset.

        Returns:
        - sampled_edges (list of tuples): The list of edges sampled within the subset.
        - updated_labels (numpy.ndarray): The updated array of vertex cluster memberships.
        """
        if n_small > len(self.available_vertices):
            raise ValueError("n_small is larger than the number of available vertices.")

        selected_vertices = np.random.choice(list(self.available_vertices), size=n_small, replace=False)
        self.available_vertices -= set(selected_vertices)

        # Update labels for selected vertices to a new cluster
        new_cluster_id = max(self.labels) + 1 if len(self.labels) > 0 else 0
        self.labels[selected_vertices] = new_cluster_id

        # Sample edges with probability r
        sampled_edges = []
        for i in range(n_small):
            for j in range(i + 1, n_small):
                if np.random.rand() < r:
                    sampled_edges.append((selected_vertices[i], selected_vertices[j]))

        return sampled_edges, self.labels

    def overlay_erdos_renyi_optimized(self, s):
        """
        Efficiently overlays the graph with unique edges based on the Erdős–Rényi model parameter s.

        Parameters:
        - s (float): Probability of an edge existing between any two vertices.

        Returns:
        - new_edges (list): A list of new, unique edges added to the graph.
        """
        N = self.graph.num_vertices()
        # Calculate the expected number of new edges to add
        total_possible_edges = N * (N - 1) // 2
        num_edges_to_add = int(np.round(s * total_possible_edges))

        new_edges = set()
        attempts = 0
        max_attempts = num_edges_to_add * 10  # Set a limit to prevent infinite loops

        while len(new_edges) < num_edges_to_add and attempts < max_attempts:
            u, v = np.random.randint(0, N, size=2)
            if u != v and not self.graph.edge(u, v):
                # Add the edge if it's not a self-loop and doesn't already exist
                new_edges.add((min(u, v), max(u, v)))  # Ensure consistency in edge direction
            attempts += 1


        return list(new_edges)


    def get_graph(self):
        """
        Returns the current state of the graph.

        Returns:
        - Graph: The current graph-tool Graph object.
        """
        return self.graph

    def get_labels(self):
        """
        Returns the current cluster memberships of all vertices.

        Returns:
        - labels (numpy.ndarray): The array of vertex cluster memberships.
        """
        return self.labels
