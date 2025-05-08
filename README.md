# `totopos` 🧫 🔃 🧬 🧪

**Totopos** - Trajectory Outlining for TOPOlogical Structures in single-cell data 

## Overview

A Python package for analyzing and visualizing topological features in single-cell data. Totopos identifies loops and other topological structures in your data, the cells that form these structures (`topoCells`), and the genes associated with them (`topoGenes`).

`totopos` can also be used for other types of data to get homology representatives, and computing the variables important variables driving a particular homology class. 

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
import totopos.genes as tpg
import totopos.cells as tpc

# Read sc data 
adata = ad.read_h5ad("path/to/sc.h5ad")

# Compute persistent homology and determine if there are topological loops
ph = ripser(adata.obsm["pcs"])

# Compute topoCells
topological_loops = tpc.critical_edge_method(adata.obsm["pcs"], ph, n_loops=1)

# Compute topoGenes
grads, topogene_scores = tpg.topological_scores_perturbation_torch_ripser(
    adata, ph, n_pcs = 20, ix_top_class = 1
)
```

For a full explanation, see the `examples` folder. 
