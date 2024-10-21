import time
import numpy as np
import networkx as nx
from collections import deque

# DBSCAN Parameters
EPSILON = 1  # Maximum distance between two samples
MIN_SAMPLES = 2  # Minimum number of samples to be a core point

# Load graph from an edge list
def load_graph_from_edgelist(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            node1, node2 = map(int, line.strip().split())
            G.add_edge(node1, node2)
    return G

# DBSCAN implementation for graph clustering
def dbscan(graph, epsilon, min_samples):
    # Initialize node labels (UNVISITED or assigned to a cluster)
    labels = {node: 'UNVISITED' for node in graph.nodes}
    cluster_id = 0

    def get_neighbors(node):
        return list(graph.neighbors(node))

    def expand_cluster(node, neighbors, cluster_id):
        labels[node] = cluster_id
        queue = deque(neighbors)

        while queue:
            current_node = queue.popleft()

            if labels[current_node] == 'UNVISITED':
                current_neighbors = get_neighbors(current_node)
                if len(current_neighbors) >= min_samples:
                    queue.extend(current_neighbors)

            if labels[current_node] == 'UNVISITED' or labels[current_node] == 'NOISE':
                labels[current_node] = cluster_id

    for node in graph.nodes:
        if labels[node] == 'UNVISITED':
            neighbors = get_neighbors(node)

            if len(neighbors) < min_samples:
                labels[node] = 'NOISE'
            else:
                cluster_id += 1
                expand_cluster(node, neighbors, cluster_id)

    return labels

# Function to evaluate clustering effectiveness using modularity
def evaluate_modularity(graph, labels):
    partition = {}
    for node, cluster in labels.items():
        if cluster not in partition:
            partition[cluster] = []
        partition[cluster].append(node)
    return nx.algorithms.community.quality.modularity(graph, partition.values())

if __name__ == "__main__":
    # Test on a small dataset
    file_path_small = "CA-GrQc.txt"  # Small dataset
    graph_small = load_graph_from_edgelist(file_path_small)

    # Effectiveness and efficiency test on small dataset
    start_time = time.time()
    labels_small = dbscan(graph_small, EPSILON, MIN_SAMPLES)
    end_time = time.time()
    running_time_small = end_time - start_time

    # Effectiveness: Report modularity as a clustering evaluation metric
    modularity_score = evaluate_modularity(graph_small, labels_small)

    print(f"Effectiveness Test on Small Dataset:")
    print(f"Modularity Score: {modularity_score}")
    print(f"Efficiency Test: Running Time: {running_time_small} seconds")

    # Test on a large dataset for scalability
    file_path_large = "com-dblp.ungraph.txt"  # Large dataset
    graph_large = load_graph_from_edgelist(file_path_large)

    # Scalability test on large dataset
    start_time = time.time()
    labels_large = dbscan(graph_large, EPSILON, MIN_SAMPLES)
    end_time = time.time()
    running_time_large = end_time - start_time

    # Effectiveness: Report modularity as a clustering evaluation metric for large dataset
    modularity_score_large = evaluate_modularity(graph_large, labels_large)

    print(f"Scalability Test on Large Dataset:")
    print(f"Modularity Score: {modularity_score_large}")
    print(f"Running Time: {running_time_large} seconds")
