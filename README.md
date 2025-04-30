# `totopos`

<img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python version">

**Totopos** - Trajectory Outlining for TOPOlogical Structures in single-cell data

## Overview

A Python package for analyzing and visualizing topological features in single-cell data. Totopos identifies loops and other topological structures in your data, the cells that form these structures (`topoCells`), and the genes associated with them (`topoGenes`).

## Installation

1. Create conda environment
    ```bash
    conda env create -f environment.yml 
    ```

2. Activate environment
    ```bash
    conda activate totopos
    ```

3. Install `pytorch` depending on your setup
    - See [PyTorch installation guide](https://pytorch.org/get-started/locally/) for reference

4. Install `totopos` module
    ```bash
    pip install -e .  # run in the repo's root directory
    ```

## Basic Usage

```python
import anndata as ad
from ripser import ripser
from persim import plot_diagrams
import totopos.genes as tpg
import totopos.cells as tpc

# Read sc data 
adata = ad.read_h5ad("path/to/sc.h5ad")

# Compute persistent homology and determine if there are topological loops
ph = ripser(adata.obsm["pcs"])

# Compute topoCells
topological_loops = tpc.critical_edge_method(
    adata.obsm["pcs], ph, n_loops = 1
)

# Compute topoGenes

grads, ranking_scores = topological_scores_perturbation_torch_ripser(
    adata, ph, n_pcs, ix_top_class = 1
)
```

For a full explanation, see the tutorials folder. 
