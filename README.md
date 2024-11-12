# DBSCAN

# Arxiv General Relativity Collaboration Network (CA-GrQc)

## Overview

This project processes the `CA-GrQc.txt` dataset, a directed collaboration network from the Arxiv General Relativity category. The dataset includes relationships between authors based on co-authorship, where an edge exists if two authors coauthored at least one paper. This code allows you to explore and analyze academic collaboration patterns using network analysis techniques.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Code](#running-the-code)
- [Example Usage](#example-usage)
- [License](#license)

## Prerequisites

To run this code, ensure you have Python 3.x installed with the following Python libraries:
- `networkx`
- `matplotlib` (optional for visualizations)
- `pandas` (if you need additional data processing)

You can install them using:

```bash
pip install networkx matplotlib pandas
```

## Installation

1. Clone this repository to your local machine.
2. Place the `CA-GrQc.txt` dataset file in the root directory of the project folder, or specify its path in the configuration file.

## Configuration

Ensure that the configuration points to the location of the dataset:

- **Dataset Path**: Update the file path for `CA-GrQc.txt` in the configuration section of the code (if applicable).
  
For example, in the Python script, it might look like this:

```python
# Configuration
DATASET_PATH = "CA-GrQc.txt"
```

## Running the Code

To process and analyze the dataset, run the main script. Here is a general command:

```bash
python main.py
```

This script reads the dataset, builds a directed graph of the collaboration network, and can perform various analyses, such as identifying influential authors or detecting communities within the network.

### Command-Line Options (if applicable)

Some versions of the script may support command-line options for custom analyses. Examples include:
- `--centrality` to calculate centrality metrics for authors.
- `--community` to perform community detection.

```bash
python main.py --centrality --community
```

### Expected Output

The output of the script may include:
- Degree distribution statistics.
- Centrality metrics for authors.
- Community detection results.
- Optional visualizations saved to the `/output` directory.

## Example Usage

Here is an example Python code snippet to load the dataset and print a summary of the graph:

```python
import networkx as nx

# Load the dataset
G = nx.read_edgelist("CA-GrQc.txt", create_using=nx.DiGraph(), nodetype=int)

# Print basic information
print(nx.info(G))
```
