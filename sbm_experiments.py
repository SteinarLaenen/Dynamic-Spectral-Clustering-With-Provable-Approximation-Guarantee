from graph_tool import Graph
import graph_tool as gt
from graph_tool.all import graph_draw, Graph
import numpy as np
import os
import random
import time
from datetime import datetime

from src.spectral_clustering import *
from src.datasets import *
from src.contractedgraph import *
from src.sparsifier import *
from src.sbm import *

import scipy
import scipy.sparse as sp

def run_experiment_merge_clusters():
    iterations = 10
    n_large = 20000
    k_large = 5
    n_small = 500
    k_small = 20
    sizes = [n_large]*5 + [n_small]*k_small
    n = sum(sizes)
    k = k_large + k_small
    p = 0.1
    q = 0.001
    r = 0.95
    s = 0.000001

    hyperparameters = f"sbm_merge_experiment_n_{n}_k_{k}_n_small_{n_small}_n_large_{n_large}_k_large_{k_large}_k_small_{k_small}_p_{p}_q_{q}_r_{r}_s_{s}_iterations_{iterations}"

    experiment_data = {
        'runtime_full_sc': [],
        'runtime_sparse_sc': [],
        'runtime_contract_sc': [],
        'FULL_ARI': [],
        'SPARSE_ARI': [],
        'CONTRACT_ARI': []
    }

    dynamic_sbm = DynamicSBMGraphMerge(k, sizes, p, q)
    true_graph, true_labels= dynamic_sbm.graph, dynamic_sbm.labels
    sparsifier = DynamicGraphSparsifier(true_graph, sampling_constant = 5)
    sparsifier.create_sparsifier()

    start_time = time.time()
    sparse_clusters, sparse_labels, ari_sparse, runtime_sparse_sc = run_spectral_clustering(sparsifier.sparsified_graph, true_labels,                                                                                        n_clusters=k)
    cg = Contracted_Graph(true_graph, sparsifier.get_sparsified_graph())
    cg.initialize(sparse_clusters, sparse_labels)

    print("initialized sparsifier and contracted graph(s):", time.time() - start_time)

    num_clusters = k

    clusters_to_merge = (k-2, k-1)


    for iteration in range(iterations):
        # cluster edges to add
        edges_to_add, true_labels = dynamic_sbm.merge_clusters(clusters_to_merge[0], clusters_to_merge[1], r)
        num_clusters -= 1
        clusters_to_merge = (clusters_to_merge[0]-2, clusters_to_merge[0]-2)

        # random edges to add
        random_edges_to_add = dynamic_sbm.overlay_erdos_renyi_optimized(s)

        # add them together
        edges_to_add = np.vstack((np.array(edges_to_add), np.array(random_edges_to_add)))

        # update full graph
        true_graph.add_edge_list(edges_to_add)
        full_clusters, full_labels, ari_full, runtime_full_sc = run_spectral_clustering(sparsifier.original_graph, true_labels,
                                                                                        n_clusters=num_clusters)
        print("full ari/time:", ari_full, runtime_full_sc)

        experiment_data['runtime_full_sc'].append(runtime_full_sc)
        experiment_data['FULL_ARI'].append(ari_full)


        # update sparse graph
        start_time = time.time()
        sparsifier.update_sparsifier(edges_to_add, verbose=False)
        update_sparse_graph_time = time.time() - start_time

        sparse_clusters, sparse_labels, ari_sparse, runtime_sparse_sc = run_spectral_clustering(sparsifier.sparsified_graph,
                                                                                                true_labels,
                                                                                                n_clusters=num_clusters)

        experiment_data['runtime_sparse_sc'].append(runtime_sparse_sc + update_sparse_graph_time)
        experiment_data['SPARSE_ARI'].append(ari_sparse)


        print("sparse ari/time:", ari_sparse, runtime_sparse_sc)

        start_time = time.time()
        cg.original_graph.add_edge_list(edges_to_add)
        cg.update(edges_to_add)
        time_update_cg = time.time() - start_time
        print("time to update cg:", time_update_cg)

        contract_clusters, contract_labels, contract_ari, runtime_contract_sc = cg.spectral_clustering_on_contracted(num_clusters, true_labels)
        print("contract ari/time:", contract_ari, runtime_contract_sc + time_update_cg)

        experiment_data['runtime_contract_sc'].append(runtime_contract_sc  + time_update_cg)
        experiment_data['CONTRACT_ARI'].append(contract_ari)

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Construct the filename with the 'results' directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f"experiment_{hyperparameters}_{timestamp}.txt")

    # Write the data to the file in the 'results' directory
    with open(filename, 'w') as file:
        for key, values in experiment_data.items():
            file.write(f"{key}: {values}\n")

def run_experiment_new_clusters():
    # parameters
    iterations = 10
    n_new = 400
    n = 30000
    k = 10
    p = 0.1
    q = 0.001
    r = 0.95
    s = 0.000001

    hyperparameters = f"sbm_new_cluster_experiment_n_{n}_k_{k}_n_new_{n_new}_p_{p}_q_{q}_r_{r}_s_{s}_iterations_{iterations}"

    experiment_data = {
        'runtime_full_sc': [],
        'runtime_sparse_sc': [],
        'runtime_contract_sc': [],
        'FULL_ARI': [],
        'SPARSE_ARI': [],
        'CONTRACT_ARI': []
    }

    dynamic_sbm = DynamicSBMGraph(k, n, p, q)
    true_graph, true_labels= dynamic_sbm.graph, dynamic_sbm.labels
    sparsifier = DynamicGraphSparsifier(true_graph, sampling_constant = 5)
    sparsifier.create_sparsifier()


    start_time = time.time()
    sparse_clusters, sparse_labels, ari_sparse, runtime_sparse_sc = run_spectral_clustering(sparsifier.sparsified_graph, true_labels,                                                                                        n_clusters=k)
    cg = Contracted_Graph(true_graph, sparsifier.get_sparsified_graph(), degree_trigger=0.8)
    cg.initialize(sparse_clusters, sparse_labels)

    print("initialized sparsifier and contracted graph:", time.time() - start_time)

    num_clusters = k

    for iteration in range(iterations):
        # cluster edges to add
        edges_to_add, true_labels = dynamic_sbm.sample_internal_edges(n_new, 1)
        num_clusters += 1

        # random edges to add
        random_edges_to_add = dynamic_sbm.overlay_erdos_renyi_optimized(s)

        # add them together
        edges_to_add = np.vstack((edges_to_add, random_edges_to_add))

        # update full graph
        true_graph.add_edge_list(edges_to_add)
        full_clusters, full_labels, ari_full, runtime_full_sc = run_spectral_clustering(sparsifier.original_graph, true_labels,
                                                                                        n_clusters=num_clusters)
        print("full ari/time:", ari_full, runtime_full_sc)

        experiment_data['runtime_full_sc'].append(runtime_full_sc)
        experiment_data['FULL_ARI'].append(ari_full)


        # update sparse graph
        start_time = time.time()
        sparsifier.update_sparsifier(edges_to_add, verbose=False)
        update_sparse_graph_time = time.time() - start_time

        sparse_clusters, sparse_labels, ari_sparse, runtime_sparse_sc = run_spectral_clustering(sparsifier.sparsified_graph,
                                                                                                true_labels,
                                                                                                n_clusters=num_clusters)

        experiment_data['runtime_sparse_sc'].append(runtime_sparse_sc + update_sparse_graph_time)
        experiment_data['SPARSE_ARI'].append(ari_sparse)


        print("sparse ari/time:", ari_sparse, runtime_sparse_sc)

        start_time = time.time()
        cg.original_graph.add_edge_list(edges_to_add)
        cg.update(edges_to_add)
        time_update_cg = time.time() - start_time
        print("time to update cg:", time_update_cg)

        contract_clusters, contract_labels, contract_ari, runtime_contract_sc = cg.spectral_clustering_on_contracted(num_clusters, true_labels)
        print("contract ari/time:", contract_ari, runtime_contract_sc + time_update_cg)

        experiment_data['runtime_contract_sc'].append(runtime_contract_sc  + time_update_cg)
        experiment_data['CONTRACT_ARI'].append(contract_ari)

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Construct the filename with the 'results' directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f"experiment_{hyperparameters}_{timestamp}.txt")

    # Write the data to the file in the 'results' directory
    with open(filename, 'w') as file:
        for key, values in experiment_data.items():
            file.write(f"{key}: {values}\n")


def compute_first_k_eigenvalues(graph, k):
    if 'edge_weight' in graph.ep.keys():
        adjacency_matrix = gt.spectral.adjacency(graph, weight=graph.ep['edge_weight'])
        laplacian_matrix = compute_normalized_laplacian(adjacency_matrix)
    else:
        laplacian_matrix = gt.spectral.laplacian(graph, norm=True, weight=None)
    # Find the bottom eigenvectors of the laplacian matrix
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(laplacian_matrix, k, which='SM')

    return eigenvalues

if __name__ == "__main__":
    # run both experiments 10 times
    for i in range(10):
        run_experiment_new_clusters()
        run_experiment_merge_clusters()
