from dreimac import CircularCoords
from persim import plot_diagrams
import numpy as np

def get_circular_coords(
    topological_loops,
    loop_index: int = 0,
    n_dim: int = None,
    plot: bool = False,
    n_landmarks: int = 1000,
    prime: int = 47,
    cocycle_idx: int = 0,
    standard_range: bool = False
):
    """
    Compute (and optionally plot) circular coordinates for a given loop in `topological_loops`.

    Parameters
    ----------
    topological_loops : list of dict
        Each dict must contain a 'topocells' key holding an array-like shape (n_samples, n_features).
    loop_index : int, default=0
        Which entry of `topological_loops` to use.
    n_dim : int, optional
        Number of leading dimensions (columns) to retain from the extracted topocells.
        If None, uses all columns.
    plot : bool, default=False
        If True, calls plot_diagrams on the persistence diagrams.
    n_landmarks : int, default=1000
        Number of landmarks for CircularCoords.
    prime : int, default=47
        Prime for finite-field arithmetic.
    cocycle_idx : int, default=0
        Which cocycle to extract coordinates from.
    standard_range : bool, default=False
        If True, rescales coords to [0,1]; otherwise returns raw angles.

    Returns
    -------
    circ_coord : ndarray, shape (n_samples,)
        The circular coordinate for each sample in the chosen loop.
    """
    # extract the topocells for this loop
    try:
        topocells = np.asarray(topological_loops[loop_index]['topocells'])
    except (IndexError, KeyError, TypeError):
        raise ValueError(f"Could not extract 'topocells' from loop_index={loop_index}")

    # validate shape
    if topocells.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {topocells.shape}")

    # determine how many dimensions to keep
    max_dim = topocells.shape[1]
    if n_dim is None:
        n_dim = max_dim
    if not isinstance(n_dim, int) or n_dim < 1 or n_dim > max_dim:
        raise ValueError(f"`n_dim` must be an integer in [1, {max_dim}], got {n_dim}")

    # slice off the first n_dim columns
    X = topocells[:, :n_dim]

    # build and (optionally) plot
    cc = CircularCoords(X, n_landmarks=n_landmarks, prime=prime)
    if plot:
        plot_diagrams(cc.dgms_)

    # compute & return
    return cc.get_coordinates(cocycle_idx=cocycle_idx, standard_range=standard_range)

import torch

def topo_scores_rep_sampling(
    adata,
    topological_loops,
    coords,
    loop_index: int = 0,
    n_reps: int = 100,
    frac_rep: float = 0.2,
    seed: int = 0,
):
    """
    Compute feature-importance scores via gradients of a topological loop loss.

    Parameters
    ----------
    adata : AnnData
        Your AnnData object whose .X is (n_cells × n_features).
    topological_loops : list of dict
        Each entry must have a 'topocell_ixs' key giving a list/array of cell indices.
    coords : array‐like, shape (n_cells_total,)
        Circular coordinate value for each cell (e.g. output of get_circular_coords).
    loop_index : int, default=0
        Which loop in `topological_loops` to use.
    n_reps : int, default=100
        How many random subsamples to average over.
    frac_rep : float ∈ (0,1], default=0.2
        Fraction of the loop’s cells to sample each repetition.
    seed : int, default=0
        RNG seed for reproducibility.

    Returns
    -------
    scores : ndarray, shape (n_features,)
        L2-norm of the gradient w.r.t. each feature.
    sorted_indices : ndarray, shape (n_features,)
        Indices of features sorted by descending `scores`.
    """
    # --- validate & extract the relevant cells ----------------
    try:
        topocell_ixs = topological_loops[loop_index]['topocell_ixs']
    except (IndexError, KeyError, TypeError):
        raise ValueError(f"Could not get 'topocell_ixs' from loop index {loop_index}")
    
    # convert adata.X to a dense NumPy array if needed
    X_full = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    topocells_full = X_full[topocell_ixs, :]  # (n_loop_cells × n_features)

    # --- set seeds and build torch tensor ---------------------
    np.random.seed(seed)
    torch.manual_seed(seed)
    pts = torch.tensor(topocells_full, dtype=torch.float32, requires_grad=True)

    # --- accumulate topological‐loss over random subsamples ----
    topo_loss_total = torch.tensor(0.0, dtype=torch.float32)
    n_loop = pts.shape[0]
    coords = np.asarray(coords)
    if coords.shape[0] != topocells_full.shape[0]:
        raise ValueError("`coords` length must match number of cells in `adata`")

    for _ in range(n_reps):
        # how many loop‐cells to pick
        n_cells = torch.randint(
            low=1, high=int(frac_rep * n_loop) + 1, size=(1,)
        ).item()
        perm = torch.randperm(n_loop)[:n_cells]
        rep_pts = pts[perm]                      # (n_cells × n_features)
        
        # order them by their circular‐coordinate
        cc_vals = coords[perm.numpy()]
        order = np.argsort(cc_vals)
        rep_pts = rep_pts[torch.LongTensor(order)]
        
        # close the loop and sum edge‐lengths
        closed = torch.cat([rep_pts, rep_pts[0:1]], dim=0)
        diffs = closed[1:] - closed[:-1]
        topo_loss_total = topo_loss_total + diffs.norm(dim=1).sum()

    # --- backprop & extract feature scores -------------------
    topo_loss_total.backward()
    grad = pts.grad                          # (n_loop × n_features)
    scores = grad.norm(dim=0).cpu().numpy()  # (n_features,)
    sorted_indices = np.argsort(scores)[::-1]

    return scores, sorted_indices


def get_topogenes(
    adata,
    feat_order,
    n_genes: int = 20,
    gene_name_col: str = 'gene_name'
) -> list:
    """
    Retrieve the top gene names from an AnnData object based on a list of
    feature indices sorted by importance.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix whose .var DataFrame contains gene metadata.
    feat_order : array-like of int
        Array of feature indices, typically sorted descending by importance
        (e.g. output of compute_topo_gradient_scores).
    n_genes : int, default=20
        Number of top genes to return.
    gene_name_col : str, default='gene_name'
        Column name in `adata.var` that holds the gene names.

    Returns
    -------
    topogenes : list of str
        The names of the top `n_genes` features.
    """
    # Validate inputs
    if not hasattr(adata, 'var'):
        raise ValueError("`adata` must be an AnnData with a .var DataFrame")
    if gene_name_col not in adata.var.columns:
        raise KeyError(f"`gene_name_col` '{gene_name_col}' not found in adata.var.columns")

    feat_order = np.asarray(feat_order, dtype=int)
    if feat_order.ndim != 1:
        raise ValueError("`feat_order` must be a one-dimensional array of feature indices")
    if n_genes < 1 or n_genes > feat_order.size:
        raise ValueError(f"`n_genes` must be between 1 and {feat_order.size}, got {n_genes}")

    # Select the top indices
    top_indices = feat_order[:n_genes]

    # Retrieve gene names
    # Use .iloc in case var is not ordered by feature index
    gene_names = adata.var[gene_name_col].iloc[top_indices].tolist()

    return gene_names
