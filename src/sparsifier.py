from graph_tool import Graph
import graph_tool as gt
from itertools import chain
import numpy as np
import random
import time
import pandas as pd


from src.spectral_clustering import *
from src.datasets import *

from sklearn.metrics import adjusted_rand_score


class DynamicGraphSparsifier:
    def __init__(self, graph: Graph, sampling_constant: int = 100) -> None:
        """
        Initialize the Graph Sparsifier for Graph-tool graphs.

        Args:
        graph (Graph): The original graph to be sparsified, from Graph-tool.
        sampling_constant (int): A constant used in the calculation of the sampling probability.
        """
        self.original_graph = graph
        self.sparsified_graph = Graph(directed=False)
        self.sampling_constant = sampling_constant


        # Initialize property maps for degrees and sampling probabilities
        self.node_degrees = self.original_graph.degree_property_map("total")

        self.degree_array = self.node_degrees.a
        self.used_samping_parameters = None

    def create_sparsifier(self):

        self.sparsified_graph = Graph(directed=False)
        self.sparsified_graph.set_fast_edge_removal(fast=True)

        # Initialize property maps for degrees and sampling probabilities
        self.node_degrees = self.original_graph.degree_property_map("total")


        self.degree_array = self.node_degrees.a


        log_n = np.log(self.original_graph.num_vertices())
        edges = self.original_graph.get_edges()

        # add vertices
        for v in self.original_graph.vertices():
            self.sparsified_graph.add_vertex()

        degrees_u = self.degree_array[edges[:, 0]]
        degrees_v = self.degree_array[edges[:, 1]]
        prob_u = np.minimum(1, self.sampling_constant * (log_n / degrees_u))
        prob_v = np.minimum(1, self.sampling_constant * (log_n / degrees_v))
        probs = prob_u + prob_v - prob_u * prob_v

        # Vectorized edge sampling
        random_values = np.random.rand(edges.shape[0])
        edges_to_add = edges[random_values < probs]
        weights = np.array([1 / probs[random_values < probs]]).T

        edges_and_weights = np.concatenate((edges_to_add, weights), axis=1)

        self.sparsified_graph.add_edge_list(edges_and_weights, eprops=[('edge_weight', 'double')])

        self.used_sampling_parameters = np.array(log_n / np.maximum(self.degree_array, 1))

    def print_verbose(self, message, start):
        print(f"{message}: {round(time.time() - start, 2)}s")

    def update_sparsifier(self, incoming_edges, verbose=True):
        def print_verbose(message, start):
            if verbose:
                print(f"{message}: {round(time.time() - start, 2)}s")

        start_time = time.time()
        n = self.original_graph.num_vertices()
        n_sparse_old = self.sparsified_graph.num_vertices()
        log_n = np.log(n)

        # Update degrees and recompute the degree array
        self.node_degrees = self.original_graph.degree_property_map("total")
        self.degree_array = self.node_degrees.a

        print_verbose("Adding nodes to sparse graph processing time", start_time)
        affected_nodes = set(chain.from_iterable(incoming_edges))

        # new_nodes = affected_nodes - set(range(self.original_graph.num_vertices()))
        for _ in range(n - n_sparse_old):
            self.sparsified_graph.add_vertex()
            self.used_sampling_parameters = np.append(self.used_sampling_parameters, log_n/self.degree_array[_])

        start_time = time.time()

        edges_to_resample = incoming_edges

        # maximum taken to deal with degree 0 vertices
        current_sampling_parameter_array = self.sampling_constant * log_n / np.maximum(self.degree_array, 1)

        used_sampling_parameter_array = self.used_sampling_parameters

        condition1 = current_sampling_parameter_array > 10 * used_sampling_parameter_array
        condition2 = current_sampling_parameter_array < 0.1 * used_sampling_parameter_array
        combined_condition = np.logical_or(condition1, condition2)
        vertices_to_resample = np.where(combined_condition)[0]

        if len(vertices_to_resample) >= int(n/100):
            # Because edge addition is much faster than edge removal in graph-tool,
            # if too many vertices have to be resampled we reconstruct the sparsifier.
            self.create_sparsifier()
            return

        if verbose:
            print("num of vertices to resample", len(vertices_to_resample))
        start_time = time.time()
        for idx in vertices_to_resample:
            vertex = self.original_graph.vertex(idx)
            self.sparsified_graph.clear_vertex(self.sparsified_graph.vertex(idx))
            adjacent_edges = self.original_graph.get_all_edges(vertex)
            edges_to_resample = np.vstack((edges_to_resample, adjacent_edges))
        # edges_to_resample = np.unique(np.sort(edges_to_resample), axis=0)

        print_verbose("clearing_vertices", start_time)
        self.used_sampling_parameters[vertices_to_resample] = log_n/np.maximum(self.degree_array[vertices_to_resample], 1)

        edges_df = pd.DataFrame(edges_to_resample, columns=['source', 'target'])

        # Ensure sorting of each row for consistent edge representation
        # This is equivalent to np.sort(old_edges_to_add[1:,:], axis=1) but for pandas DataFrame
        edges_df = pd.DataFrame(np.sort(edges_df.values, axis=1), columns=edges_df.columns)

        # Drop duplicates to get unique edges
        unique_edges_df = edges_df.drop_duplicates()

        # Convert back to numpy array if needed (optional step depending on further usage)
        unique_edges = unique_edges_df.to_numpy()
        edges_to_resample = unique_edges

        # edges_to_resample = np.unique(np.sort(edges_to_resample), axis=0)

        print_verbose("get_unique edge list", start_time)

        # Compute sampling probabilities for edges to resample
        start_time = time.time()
        degrees_u = self.degree_array[edges_to_resample[:, 0]]
        degrees_v = self.degree_array[edges_to_resample[:, 1]]
        prob_u = np.minimum(1, self.sampling_constant * (log_n / np.maximum(degrees_u, 1)))
        prob_v = np.minimum(1, self.sampling_constant * (log_n / np.maximum(degrees_v, 1)))
        probs = prob_u + prob_v - prob_u * prob_v

        # Vectorized edge sampling
        random_values = np.random.rand(edges_to_resample.shape[0])
        edges_to_add_indices = random_values < probs
        edges_to_add = edges_to_resample[edges_to_add_indices]

        # Assign weights to the edges based on their sampling probabilities
        weights = 1 / probs[edges_to_add_indices]
        print_verbose("Sampling and weight calculation", start_time)

        weights = np.array([1 / probs[random_values < probs]]).T

        edges_to_add_with_weights = np.concatenate((edges_to_add, weights), axis=1)
        start_time = time.time()

        # Add sampled edges to the sparsified graph
        self.sparsified_graph.add_edge_list(edges_to_add_with_weights, eprops=[self.sparsified_graph.ep["edge_weight"]])

        print_verbose("Edge addition to sparsified graph", start_time)


    def get_sparsified_graph(self) -> Graph:
        """
        Return the sparsified graph.

        Returns:
        Graph: The sparsified graph from Graph-tool.
        """
        return self.sparsified_graph
