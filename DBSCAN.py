#DBSCAN Project
import time
from collections import deque, defaultdict

# DBSCAN Parameters
EPSILON_SMALL = 2  # Adjusted for small dataset: consider direct and next-nearest neighbors
EPSILON_LARGE = 0.5  # For large dataset: only direct neighbors
MIN_SAMPLES_SMALL = 8  # Tuned for small dataset
MIN_SAMPLES_LARGE = 20  # Tuned for large dataset

# Load graph from an edge list
def load_graph_from_edgelist(file_path):
    graph = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            node1, node2 = map(int, line.strip().split())
            graph[node1].append(node2)
            graph[node2].append(node1)  # Since it's an undirected graph
    return graph

# DBSCAN implementation for graph clustering
def dbscan(graph, epsilon, min_samples):
    labels = {node: 'UNVISITED' for node in graph}
    cluster_id = 0

    def get_neighbors(node):
        return graph[node]

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

    for node in graph:
        if labels[node] == 'UNVISITED':
            neighbors = get_neighbors(node)

            if len(neighbors) < min_samples:
                labels[node] = 'NOISE'
            else:
                cluster_id += 1
                expand_cluster(node, neighbors, cluster_id)

    return labels

# Improved output for cluster sizes
def evaluate_cluster_sizes(labels):
    cluster_sizes = defaultdict(int)
    for node, cluster_id in labels.items():
        if cluster_id != 'NOISE':
            cluster_sizes[cluster_id] += 1
    return cluster_sizes

# Function to print the clusters in a more readable format
def print_cluster_sizes(cluster_sizes, max_clusters=10):
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Number of clusters: {len(sorted_clusters)}")
    print("Top clusters:")
    for i, (cluster_id, size) in enumerate(sorted_clusters[:max_clusters]):
        print(f"  Cluster {cluster_id}: {size} nodes")
    
    if len(sorted_clusters) > max_clusters:
        print(f"...and {len(sorted_clusters) - max_clusters} more clusters.")

if __name__ == "__main__":
    # Test on the small dataset
    file_path_small = "CA-GrQc.txt"  # Small dataset
    graph_small = load_graph_from_edgelist(file_path_small)

    # Effectiveness and efficiency test on small dataset
    start_time = time.time()
    labels_small = dbscan(graph_small, EPSILON_SMALL, MIN_SAMPLES_SMALL)
    end_time = time.time()
    running_time_small = end_time - start_time

    # Effectiveness: Report cluster sizes
    cluster_sizes_small = evaluate_cluster_sizes(labels_small)
    
    print(f"Effectiveness Test on Small Dataset (CA-GrQc.txt):")
    print_cluster_sizes(cluster_sizes_small)
    print(f"Efficiency Test: Running Time: {running_time_small:.5f} seconds\n")

    # Test on the large dataset for scalability
    file_path_large = "com-dblp.ungraph.txt"  # Large dataset
    graph_large = load_graph_from_edgelist(file_path_large)

    # Scalability test on large dataset
    start_time = time.time()
    labels_large = dbscan(graph_large, EPSILON_LARGE, MIN_SAMPLES_LARGE)
    end_time = time.time()
    running_time_large = end_time - start_time

    # Effectiveness: Report cluster sizes for large dataset
    cluster_sizes_large = evaluate_cluster_sizes(labels_large)

    print(f"Scalability Test on Large Dataset (com-dblp.ungraph.txt):")
    print_cluster_sizes(cluster_sizes_large, max_clusters=15)  # Limit large dataset output to top 15 clusters
    print(f"Running Time: {running_time_large:.5f} seconds")
