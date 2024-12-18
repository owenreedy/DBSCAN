# DBSCAN Project

This project implements the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm for clustering graph data, designed for analyzing both large and small datasets. DBSCAN clusters nodes in an undirected graph based on neighborhood density, with adjustable parameters (`epsilon` for neighborhood radius and `min_samples`) to optimize clustering based on dataset size and density.

The project includes clustering on two datasets:
- **`CA-GrQc.txt`**: A smaller dataset, suitable for quicker clustering with broader neighborhoods.
- **`com-dblp.ungraph.txt`**: A larger dataset representing a scientific collaboration network. This dataset tests the scalability and efficiency of DBSCAN on a denser, real-world graph.

## Project Structure

- **`dbscan.py`** - The main script containing the DBSCAN clustering implementation and utilities to load data, compute clusters, and evaluate cluster sizes.

## Requirements

- Python 3.x
- Standard Python libraries: `time`, `collections`, and `defaultdict` from `collections`.

No external libraries are required.

## Datasets

This project uses two datasets:
- **Small Dataset**: `CA-GrQc.txt` - Represents a co-authorship network in a small scientific field.
- **Large Dataset**: `com-dblp.ungraph.txt` - A more complex co-authorship network from the DBLP dataset, representing connections across a broader scientific community.

These files should be edge lists in the format:

```
# Commented line (optional)
node1 node2
node1 node3
...
```

Each line represents an undirected edge between `node1` and `node2`.

### Expected File Structure

Ensure your file structure is as follows:

```
project_folder/
│
├── dbscan.py
├── CA-GrQc.txt          # Small dataset
└── com-dblp.ungraph.txt  # Large dataset
```

## Running the Project

To run the DBSCAN clustering, execute the following command in the terminal:

```bash
python dbscan.py
```

This will load the datasets, apply DBSCAN with tuned parameters for each dataset, and display the results.

### Output

The script will print:
- **Cluster sizes**: Number of nodes in each cluster, showing the top clusters.
- **Execution time**: Time taken for clustering each dataset.

## Configuration

You can adjust the parameters for epsilon (`EPSILON_SMALL`, `EPSILON_LARGE`) and minimum samples (`MIN_SAMPLES_SMALL`, `MIN_SAMPLES_LARGE`) directly in the script to tune the clustering for different datasets.

```python
# DBSCAN Parameters
EPSILON_SMALL = 2      # Epsilon for small dataset
EPSILON_LARGE = 0.5    # Epsilon for large dataset
MIN_SAMPLES_SMALL = 8  # Minimum samples for small dataset
MIN_SAMPLES_LARGE = 20 # Minimum samples for large dataset
```

Adjust these values based on the dataset size and clustering density you aim to achieve.

## Evaluation

The clustering results are evaluated by:
- **Number of clusters**
- **Size of each cluster** (Top clusters are printed for readability)

## Example

Below is an example output:

```
Effectiveness Test on Small Dataset (CA-GrQc.txt):
Number of clusters: 15
Top clusters:
  Cluster 1: 134 nodes
  Cluster 2: 98 nodes
  ...

Efficiency Test: Running Time: 0.04567 seconds

Scalability Test on Large Dataset (com-dblp.ungraph.txt):
Number of clusters: 120
Top clusters:
  Cluster 1: 1584 nodes
  Cluster 2: 1467 nodes
  ...

Running Time: 5.23489 seconds
```

