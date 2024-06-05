from graph_tool import Graph
import pandas as pd
import graph_tool as gt
import numpy as np
import os
import time

from src.spectral_clustering import *
from src.datasets import *
from src.contractedgraph import *
from src.sparsifier import *

from sklearn.metrics import adjusted_rand_score

class Contracted_Graph:
    def __init__(self, graph, sparsifier, degree_trigger=0.5):
        self.original_graph = graph  # The original graph-tool graph
        self.sparsified_graph = sparsifier  # The sparsifier used to construct the graph at initialisation
        self.contracted_graph = Graph(directed=False)  # Initialize an undirected graph for the contracted version
        self.cluster_map = {}  # Maps each vertex to its cluster
        self.cluster_vertices = {}  # Maps each cluster to its new vertex in the contracted graph
        self.degree_trigger = degree_trigger


        # useful arrays
        self.idx_to_og_vertex_array = np.array([vertex for vertex in self.sparsified_graph.vertices()])
        self.idx_to_og_idx_array = self.sparsified_graph.get_vertices()

        # this will be a list of lists, where the indices in the sublist correspond to the vertices
        # from the uncontracted graph that map to the contracted graph
        self.contracted_node_to_full_node_map = []
        self.contract_idx_to_vertex_label = []

    def initialize(self, clusters, labels):
        """
        Initialises the contracted graph based on the provided clustering.
        Each cluster is represented by a single vertex in the contracted graph,
        and edges represent the total weight of connections between clusters.

        Parameters:
        - clustering: list of lists, where every sublist are all the vertex idx in the same cluster
        """

        self.og_num_contract_nodes = len(clusters)

        # first, ensure that the indices returned by spectral clustering actually
        # correspond to valid indexes in the original graph
        for cluster in clusters:
            reindexed_cluster = []
            for elem in cluster:
                reindexed_element = self.idx_to_og_idx_array[elem]
                reindexed_cluster.append(reindexed_element)
            # use set representation for fast vertex removal
            self.contracted_node_to_full_node_map.append(set(reindexed_cluster))


        start_time = time.time()

        edges_sparsified_graph = self.sparsified_graph.get_edges([self.sparsified_graph.ep["edge_weight"]])

        # Example data, ensuring integer types explicitly
        idx_to_og_idx_array = self.idx_to_og_idx_array  # Original vertex labels
        cluster_assignments = labels  # Cluster assignments by index in idx_to_og_idx_array

        # Direct mapping from original vertex indices to clusters
        vertex_index_to_cluster_mapping = np.full(idx_to_og_idx_array.max() + 1, -1, dtype=int)
        vertex_index_to_cluster_mapping[idx_to_og_idx_array] = cluster_assignments

        # Transforming edge list vertex indices to cluster indices, including handling of zero-weight edges
        clusters_source = vertex_index_to_cluster_mapping[edges_sparsified_graph[:, 0].astype(int)]
        clusters_target = vertex_index_to_cluster_mapping[edges_sparsified_graph[:, 1].astype(int)]

        # Aggregating edge weights by cluster pairs
        cluster_pairs_and_weights = np.concatenate((clusters_source[:, None], clusters_target[:, None], edges_sparsified_graph[:, 2][:, None]), axis=1)
        unique_pairs, indices = np.unique(cluster_pairs_and_weights[:, :2].astype(int), axis=0, return_inverse=True)
        total_weights = np.zeros(unique_pairs.shape[0], dtype=float)
        np.add.at(total_weights, indices, cluster_pairs_and_weights[:, 2])

        # Assuming 'num_clusters' is the total number of unique clusters
        num_clusters = len(clusters)

        # Generate all possible pairs of clusters (including self-pairs for undirected graphs)
        all_cluster_pairs = np.array([[i, j] for i in range(num_clusters) for j in range(num_clusters)])

        # Initialize an array for weights of all cluster pairs, defaulting to zero
        all_weights = np.zeros(len(all_cluster_pairs))

        # Map from unique_pairs to their indices for quick lookup
        pair_to_index = {tuple(pair): i for i, pair in enumerate(unique_pairs)}

        # Populate the weights for existing pairs
        for i, pair in enumerate(all_cluster_pairs):
            if tuple(pair) in pair_to_index:
                all_weights[i] = total_weights[pair_to_index[tuple(pair)]]

        # Prepare an edge property map for weights in the contracted graph
        edge_weight = self.contracted_graph.new_edge_property("double")

        # Iterate over each pair and add an edge with the corresponding weight
        for pair, weight in zip(all_cluster_pairs, all_weights):
            source_cluster, target_cluster = pair
            # Add edge between clusters
            e = self.contracted_graph.add_edge(source_cluster,
                                               target_cluster)
            # Set the edge weight
            if source_cluster == target_cluster:
                edge_weight[e] = weight
            else:
                edge_weight[e] = weight

        # Attach the edge property map to the contracted graph
        self.contracted_graph.edge_properties["edge_weight"] = edge_weight


        # set the degrees that were used to construct the contracted graph
        self.degree_array = self.original_graph.degree_property_map("total").a

        # store this array, which is needed to keep track of what vertices are pulled out
        self.degree_array_at_construction = self.degree_array

        return


    def compute_weight_between_clusters(self, edges, og_vertex_to_contract_idx):
        """
        Given a set of edges, this function computes the representative edge weight
        updates in the contracted graph using pandas for faster unique row identification.
        """
        # Reindex edges to contracted graph cluster indices
        reindexed_source = og_vertex_to_contract_idx[edges[:, 0].astype(int)]
        reindexed_target = og_vertex_to_contract_idx[edges[:, 1].astype(int)]
        # Add a one column to represent initial weights between edges
        reindexed_edges = np.column_stack((reindexed_source, reindexed_target, np.ones(len(reindexed_source), dtype=int)))

        start_time = time.time()

        # Convert to DataFrame for easier manipulation with pandas
        df_edges = pd.DataFrame(reindexed_edges, columns=['source', 'target', 'weight'])

        # Group by source and target to find unique pairs and sum their weights
        grouped = df_edges.groupby(['source', 'target']).sum().reset_index()

        # Extract the unique pairs and total weights
        unique_pairs = grouped[['source', 'target']].values
        total_weights = grouped['weight'].values

        return unique_pairs, total_weights


    def update(self, edges_to_add):
        start_time = time.time()

        # old number of vertices
        n_old = len(self.degree_array)

        # new number of vertices
        n_new = self.original_graph.num_vertices()

        # store old degree array and old number of vertices
        n_at_construction = len(self.degree_array_at_construction)

        # get the new degree values
        self.degree_array = self.original_graph.degree_property_map("total").a


        # detect how much the degrees have changed of every vertex
        degree_change_old_vertices = self.degree_array_at_construction / np.maximum(1, self.degree_array[:n_at_construction])

        # find which vertices need to be pulled out of the contracted graph
        # we filter for less than one, as edges that have been pulled out have degree -1
        old_vertices_to_pull_out = np.where((0 < degree_change_old_vertices) & (degree_change_old_vertices < self.degree_trigger))[0].astype(int)

        # set degrees to -1 for vertices that have been pulled out, as they don't need to be
        # pulled out anymore.
        self.degree_array_at_construction[old_vertices_to_pull_out] = -1

        old_vertices_to_pull_out_set = set(old_vertices_to_pull_out)

        # now get the edges adjacent to old vertices, which are to be added to the contracted graph
        old_edges_to_add = np.array([[0,0]])
        for vertex_idx in old_vertices_to_pull_out:
            vertex = self.original_graph.vertex(vertex_idx)
            adjacent_edges = self.original_graph.get_all_edges(vertex)
            old_edges_to_add = np.vstack((old_edges_to_add, adjacent_edges))

        # remove first row which was just to do the above construction, and only get unique edges
        # old_edges_to_add = np.unique(np.sort(old_edges_to_add[1:,:], axis=1), axis=0)

        edges_df = pd.DataFrame(old_edges_to_add[1:, :], columns=['source', 'target'])

        # Ensure sorting of each row for consistent edge representation
        # This is equivalent to np.sort(old_edges_to_add[1:,:], axis=1) but for pandas DataFrame
        edges_df = pd.DataFrame(np.sort(edges_df.values, axis=1), columns=edges_df.columns)

        # Drop duplicates to get unique edges
        unique_edges_df = edges_df.drop_duplicates()

        # Convert back to numpy array if needed (optional step depending on further usage)
        unique_edges = unique_edges_df.to_numpy()


        # get the cluster assignment of each vertex.
        og_vertex_to_contract_idx = np.array(clusters_to_labels(self.contracted_node_to_full_node_map,
                                                                num_data_points=n_new))

        start_time = time.time()
        # get the total number of edges that have been removed from inter supernode weights
        edges_between_clusters, total_weights = self.compute_weight_between_clusters(old_edges_to_add,
                                                                                     og_vertex_to_contract_idx)
        for i, pair in enumerate(edges_between_clusters):
            source, target = pair[0], pair[1]
            # we add the extra check for < 0, because we have also computed the weight
            # between new vertices and vertices to be removed when calling
            # self.original_graph.vertex() above
            if not (source < 0 or target < 0) and (source < self.og_num_contract_nodes and target < self.og_num_contract_nodes):
                edge = self.contracted_graph.edge(source, target)

                # decrease edge weight based on edges that are pulled output
                self.contracted_graph.ep["edge_weight"][edge] -= total_weights[i]

                # in case edge weight gets set to zero (due to probabilistic nature of sparsifier)
                self.contracted_graph.ep["edge_weight"][edge] = max(0, self.contracted_graph.ep["edge_weight"][edge])


        # after updating the inter cluster weights, remove the pulled out vertices from their assignment in the
        updated_map = []
        for cluster in self.contracted_node_to_full_node_map:
            updated_map.append(cluster - old_vertices_to_pull_out_set)

        for vertex in old_vertices_to_pull_out:
            updated_map.append(set([vertex]))

        # add new nodes, which we identify by their index and add directly as a cluster
        for j in range(n_old, n_new):
            updated_map.append(set([j]))


        # now update the inter supernode weights
        self.contracted_node_to_full_node_map = updated_map


        # create a mapping from the vertex id in the original graph to the vertex id in the contracted graph
        og_vertex_to_contract_idx = np.array(clusters_to_labels(self.contracted_node_to_full_node_map,
                                                                        num_data_points=len(self.degree_array)))

        edges_to_add = np.array(edges_to_add)

        # these are all the edges to be added into the contracted graph.
        # edges here are still indexed by their new labels
        edges_to_add = np.vstack((edges_to_add, old_edges_to_add))


        # compute the weight needed to be added between all vertices in the contracted graph
        edges_in_cg, total_weights = self.compute_weight_between_clusters(edges_to_add,
                                                                           og_vertex_to_contract_idx)


        # add edges to contracted graph
        edges_and_weights = np.hstack((edges_in_cg[:, :2], total_weights[:, None]))

        self.contracted_graph.add_edge_list(edges_and_weights,
                                            eprops=[self.contracted_graph.ep["edge_weight"]])

        return

    def spectral_clustering_on_contracted(self, num_clusters, true_labels):
        start_time = time.time()
        not_isolated = self.contracted_graph.new_vertex_property("bool")

        # remove isolated vertices in the sparsified graph
        for v in self.contracted_graph.vertices():
            not_isolated[v] = v.out_degree() + v.in_degree() > 0

            # Set the graph's vertex filter to hide isolated vertices
        self.contracted_graph.set_vertex_filter(not_isolated)

        # run spectral clustering
        clusters = spectral_clustering(self.contracted_graph, num_clusters)

        # remap the returned clusters to their original vertices
        reindexed_clusters = []
        for cluster in clusters:
            reindexed_cluster = []
            for elem in cluster:
                reindexed_cluster += self.contracted_node_to_full_node_map[elem]

            reindexed_clusters.append(reindexed_cluster)

        reindexed_labels = clusters_to_labels(reindexed_clusters)

        end_time = time.time()
        runtime = end_time - start_time

        if len(true_labels) > len(reindexed_labels):
            true_labels = true_labels[:len(reindexed_labels)]
        if len(true_labels) < len(reindexed_labels):
            reindexed_labels = reindexed_labels[:len(true_labels)]

        ari = adjusted_rand_score(true_labels, reindexed_labels)
        self.contracted_graph.set_vertex_filter(None)

        return reindexed_clusters, reindexed_labels, ari, runtime
