from graph_tool import Graph
import graph_tool as gt
from itertools import chain
import numpy as np
import os
import random
import time
from datetime import datetime

from src.spectral_clustering import *
from src.datasets import *
from src.contractedgraph import *
from src.sparsifier import *


def update_and_cluster(knn_manager, sparsifier, current_classes, vertices_to_add, current_vertices, experiment_data, num_batches):
    """Updates the graph with new vertices, performs spectral clustering, and updates the sparsifier."""

    not_isolated = sparsifier.original_graph.new_vertex_property("bool")
    sparsified_graph = sparsifier.get_sparsified_graph()

    # remove isolated vertices in the sparsified graph
    for v in sparsified_graph.vertices():
        not_isolated[v] = v.out_degree() + v.in_degree() > 0

    # Set the graph's vertex filter to hide isolated vertices
    sparsified_graph.set_vertex_filter(not_isolated)

    subgraph_labels = []
    for v in sparsified_graph.vertices():
        old_index = sparsifier.original_graph.vertex_properties["original_indices"][v]
        subgraph_labels.append(knn_manager.graph.vertex_properties["labels"][old_index])

    start_time = time.time()
    # run spectral clustering on sparse graph
    sparse_clusters, sparse_labels, ari, runtime_sparse_sc = run_spectral_clustering(sparsified_graph, subgraph_labels, n_clusters=len(current_classes)-1)


    cg = Contracted_Graph(sparsifier.original_graph, sparsified_graph, degree_trigger=0.2)
    cg.initialize(sparse_clusters, sparse_labels)
    contract_graph_time = time.time() - start_time
    print(f"Time to contract graph (s):{contract_graph_time:.3g}")

    sparsified_graph.set_vertex_filter(None)


    # Get edges that need to be added between current vertices and new vertices.
    edges_to_add = knn_manager.get_edges_between_sets(current_vertices, vertices_to_add)

    # ensure no parallel edges are added
    edges_to_add = np.sort(edges_to_add, axis=1)
    edges_to_add = np.unique(edges_to_add, axis=0)


    # Retrieve a list of original indices of all vertices already added to the 'original_graph'.
    vertices_already_added = [
        sparsifier.original_graph.vp["original_indices"][u] for u in sparsifier.original_graph.vertices()
    ]

    # Identify all unique vertices (by their original indices) that are touched by the new edges.
    # This includes both ends of each edge.
    touched_vertices = set(chain.from_iterable(edges_to_add))

    # Determine which of these touched vertices are not yet added to the graph by subtracting
    # the set of already added vertices from the set of touched vertices.
    new_vertices = touched_vertices - set(vertices_already_added)

    # Add new vertices to the graph. For each new vertex, we also update the 'original_indices'
    # property map to keep track of the vertex's original index.
    for new_vertex in new_vertices:
        new_vertex_idx = sparsifier.original_graph.add_vertex()
        sparsifier.original_graph.vp["original_indices"][new_vertex_idx] = new_vertex

    # Find the maximum node label to define the size of the index mapping array.
    # This is necessary to map old indices to new indices correctly.
    max_node_label = max(max(vertices_already_added), max(new_vertices))

    # Create an array to map old vertex indices to new vertex indices in the graph.
    # This array will have a size of 'max_node_label + 1' to accommodate all possible indices.
    old_idx_to_new_idx = np.zeros(max_node_label + 1, dtype=int)

    # Populate the mapping array with the new vertex indices.
    for new_idx in sparsifier.original_graph.vertices():
        old_idx = sparsifier.original_graph.vp["original_indices"][new_idx]
        old_idx_to_new_idx[old_idx] = int(new_idx)

    # Reindex the edges to be added to use the new vertex indices.
    # This step is necessary because the edges were initially indexed according to the original graph,
    # and we need to translate these to the current indices in the sparsifier's graph.
    u_1 = old_idx_to_new_idx[edges_to_add[:, 0]]  # Map the first vertex of each edge
    u_2 = old_idx_to_new_idx[edges_to_add[:, 1]]  # Map the second vertex of each edge
    edges_to_add = np.vstack((u_1, u_2)).T  # Combine the mapped indices into a new array of edges


    # Randomize the edges_to_add list
    np.random.shuffle(edges_to_add)

    num_edges_to_add = len(edges_to_add)
    batch_size = num_edges_to_add // num_batches

    for batch_idx in range(num_batches):
        start_edge_idx = batch_idx * batch_size
        end_edge_idx = (batch_idx + 1) * batch_size if batch_idx < num_batches - 1 else num_edges_to_add

        batch_edges_to_add = edges_to_add[start_edge_idx:end_edge_idx]

        # add edges to the full graph
        start_time = time.time()
        sparsifier.original_graph.add_edge_list(batch_edges_to_add)

        update_full_graph_time = time.time() - start_time

        # remove isolated vertices in the graph
        not_isolated = sparsifier.original_graph.new_vertex_property("bool")

        for v in sparsifier.original_graph.vertices():
            not_isolated[v] = v.out_degree() + v.in_degree() > 0

        # Set the graph's vertex filter to hide isolated vertices
        sparsifier.original_graph.set_vertex_filter(not_isolated)
        current_vertices.extend(vertices_to_add)

        print(f"Batch {batch_idx + 1}/{num_batches}: Classes: {current_classes}, n = {sparsifier.original_graph.num_vertices()}")
        print(f"full graph update (graph): {update_full_graph_time:.3g}s. m = {sparsifier.original_graph.num_edges()}. n = {sparsifier.original_graph.num_vertices()}.")

        # get labels for spectral clustering
        subgraph_labels = []
        for v in sparsifier.original_graph.vertices():
            old_index = sparsifier.original_graph.vertex_properties["original_indices"][v]
            subgraph_labels.append(knn_manager.graph.vertex_properties["labels"][old_index])

        # perform spectral clustering on the full matrix
        full_clusters, full_labels, full_ari, runtime_full_sc = run_spectral_clustering(sparsifier.original_graph, subgraph_labels,
                                                       n_clusters=len(current_classes))
        print(f"FULL ARI: {full_ari:.3g}, Runtime SC full_graph: {runtime_full_sc:.3g}s. Total runtime = {runtime_full_sc + update_full_graph_time:.3g}s")

        experiment_data['runtime_full_sc'].append(runtime_full_sc + update_full_graph_time)
        experiment_data['FULL_ARI'].append(full_ari)

        # # unset the filter
        sparsifier.original_graph.set_vertex_filter(None)

        start_time = time.time()
        # dynamically update the cluster preserving sparsifier
        sparsifier.update_sparsifier(batch_edges_to_add, verbose=False)
        update_sparse_graph_time = time.time() - start_time
        sparsified_graph = sparsifier.get_sparsified_graph()

        # remove isolated vertices in the sparsified graph
        for v in sparsified_graph.vertices():
            not_isolated[v] = v.out_degree() + v.in_degree() > 0

        # Set the graph's vertex filter to hide isolated vertices
        sparsified_graph.set_vertex_filter(not_isolated)

        print(f"sparse graph update (graph): {update_sparse_graph_time:.3g}s. m = {sparsified_graph.num_edges()}. n = {sparsified_graph.num_vertices()}.")

        subgraph_labels = []
        for v in sparsified_graph.vertices():
            old_index = sparsifier.original_graph.vertex_properties["original_indices"][v]
            subgraph_labels.append(knn_manager.graph.vertex_properties["labels"][old_index])

        # run spectral clustering on sparse graph
        sparse_clusters, sparse_labels, sparse_ari, runtime_sparse_sc = run_spectral_clustering(sparsified_graph, subgraph_labels, n_clusters=len(current_classes))
        print(f"SPARSE ARI: {sparse_ari:.3g}, Runtime SC sparse_graph: {runtime_sparse_sc:.3g}s. Total runtime = {runtime_sparse_sc + update_full_graph_time + update_sparse_graph_time:.3g}s")


        experiment_data['runtime_sparse_sc'].append(runtime_sparse_sc + update_full_graph_time + update_sparse_graph_time)
        experiment_data['SPARSE_ARI'].append(sparse_ari)

        # cg.original_graph.add_edge_list(batch_edges_to_add)
        start_time = time.time()
        cg.update(batch_edges_to_add)
        update_contract_graph_time = time.time() - start_time
        print(f"contract graph update (graph): {update_contract_graph_time:.3g}s. m_contract = {cg.contracted_graph.num_edges()}. n_contract = {cg.contracted_graph.num_vertices()}")


        contract_clusters, contract_labels, contract_ari, runtime_contract_sc = cg.spectral_clustering_on_contracted(len(current_classes), subgraph_labels)

        print(f"CONTRACT ARI: {contract_ari:.3g}, Runtime SC contract_graph: {runtime_contract_sc:.3g}s. Total runtime = {runtime_contract_sc + update_full_graph_time + update_contract_graph_time + contract_graph_time:.3g}s \n")


        experiment_data['runtime_contract_sc'].append(runtime_contract_sc + update_full_graph_time + update_contract_graph_time + contract_graph_time)
        experiment_data['CONTRACT_ARI'].append(contract_ari)

        contract_graph_time = 0



        sparsified_graph.set_vertex_filter(None)



def randomize_dataset_classes(dataset_name, initial_class_count, group_sizes):
    """
    Randomly selects a specified number of initial classes for a given dataset
    and distributes the remaining classes into sublists of specified sizes.

    Parameters:
    - dataset_name (str): The name of the dataset. Supported values are 'mnist' and 'emnist'.
    - initial_class_count (int): The number of classes to randomly select as initial classes.
    - group_sizes (list of int): Allowed sizes for the sublists to distribute the remaining classes.

    Returns:
    - tuple: A tuple containing the list of initial classes and a list of lists representing the
      classes to add, where each sublist contains a number of classes as per the specified group sizes.
    """

    if dataset_name == 'mnist':
        total_classes = [str(i) for i in range(10)]  # MNIST has classes from 0 to 9
    elif dataset_name == 'emnist':
        total_classes = [str(i) for i in range(26)]  # EMNIST letters has classes from 0 to 25
    else:
        raise ValueError("Unsupported dataset. Choose 'mnist' or 'emnist'.")

    # Randomly select initial classes
    current_classes = random.sample(total_classes, initial_class_count)

    # Determine the remaining classes
    remaining_classes = [cls for cls in total_classes if cls not in current_classes]

    # Randomly distribute remaining classes into sublists of allowed sizes
    classes_to_add = []
    while remaining_classes:
        group_size = random.choice(group_sizes) if remaining_classes else 0
        if group_size == 0 or len(remaining_classes) < min(group_sizes):
            classes_to_add.append(remaining_classes)
            break
        group = random.sample(remaining_classes, min(group_size, len(remaining_classes)))
        classes_to_add.append(group)
        remaining_classes = [cls for cls in remaining_classes if cls not in group]

    return current_classes, classes_to_add


def test_graph_sparsification(dataset_name='mnist', k=20, num_batches = 8, ipc=None):
    """Test graph sparsification on a specified dataset."""

    experiment_data = {
        'runtime_full_sc': [],
        'runtime_sparse_sc': [],
        'runtime_contract_sc': [],
        'FULL_ARI': [],
        'SPARSE_ARI': [],
        'CONTRACT_ARI': []
    }

    knn_manager = KNN_Graph_Manager(k=k, dataset_name=dataset_name, images_per_class=ipc)
    knn_manager.build_or_load_graph()


    if dataset_name == 'mnist':
        current_classes, classes_to_add = randomize_dataset_classes('mnist', 4, [1])
    elif dataset_name == 'emnist':
        current_classes, classes_to_add = randomize_dataset_classes('emnist', 4, [2])

    initial_graph, current_vertices = build_initial_graph(knn_manager, current_classes)

    start_time = time.time()
    sparsifier = DynamicGraphSparsifier(initial_graph, sampling_constant = 1)
    sparsifier.create_sparsifier()

    print(f"Construction sparsifier: {time.time() - start_time}s")

    for cls in classes_to_add:
        current_classes = current_classes + cls
        vertices_to_add = knn_manager.get_vertices_by_labels(cls)
        update_and_cluster(knn_manager, sparsifier, current_classes,
                           vertices_to_add, current_vertices, experiment_data,
                           num_batches,
                           )

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Construct the filename with the 'results' directory
    hyperparameters = f"k_{k}_dataset_{dataset_name}_num_batches_{num_batches}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f"experiment_{hyperparameters}_{timestamp}.txt")

    # Write the data to the file in the 'results' directory
    with open(filename, 'w') as file:
        for key, values in experiment_data.items():
            file.write(f"{key}: {values}\n")

if __name__ == "__main__":
    for i in range(10):
        test_graph_sparsification('mnist', k=200, num_batches=10, ipc=None)
        test_graph_sparsification('emnist', k=100, num_batches=10, ipc=None)
