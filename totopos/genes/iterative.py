import torch
import numpy as np
from anndata import AnnData
from ..utils.utils import randomized_pca_torch

def compute_topological_scores_iterative(
    adata: AnnData, 
    n_reps: int = 100, 
    pca: bool = True, 
    n_pcs: int = 20,
    cc_key: str = 'cc',
    verbose: bool = False
) -> tuple:
    """
    Computes topological feature scores by repeatedly sampling small subsets of cells
    and measuring cyclic trajectory distances (e.g., cell cycle ring closure).
    
    Parameters
    ----------
    adata : AnnData
        Annotated gene expression matrix (cells x genes).
    n_reps : int
        Number of random subsamples to draw.
    pca : bool
        Whether to apply PCA to the data first.
    n_pcs : int
        Number of principal components to use if PCA is applied.
    cc_key : str
        Key in adata.obs containing the cyclic coordinate (e.g., cell cycle phase).
    verbose : bool
        Whether to print debugging info.
        
    Returns
    -------
    scores : np.ndarray
        Norm of gradient per gene.
    grads : torch.Tensor
        Full gradient matrix of shape (n_cells, n_genes).
    """
    data = adata.X.A
    pts = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    if pca:
        pcs = randomized_pca_torch(pts, n_pcs)
    else:
        pcs = pts

    topo_loss_total = 0.0
    obs_cc = adata.obs[cc_key].values

    for _ in range(n_reps):
        n_cells = torch.randint(2, max(3, int(0.2 * pcs.shape[0])), (1,)).item()
        sample_indices = torch.randperm(pcs.shape[0])[:n_cells]
        rep_cells = pcs[sample_indices]

        cc_values = obs_cc[sample_indices]
        ordered_indices = np.argsort(cc_values)
        ordered_rep_cells = rep_cells[ordered_indices]

        ordered_rep_cells = torch.cat([ordered_rep_cells, ordered_rep_cells[0].unsqueeze(0)], dim=0)
        dist_vecs = ordered_rep_cells[1:] - ordered_rep_cells[:-1]
        norms = dist_vecs.norm(dim=1)
        topo_loss_total += norms.sum()

    topo_loss_total.backward()
    grads = pts.grad
    scores = grads.norm(dim=0).detach().numpy()

    if verbose:
        print("Topological scoring via cycle closure completed.")

    return scores, grads


def iterative(pcs, n_reps = 100, ):
    
    topo_loss_total = 0

    for _ in range(n_reps):
        n_cells = torch.randint(0, int(0.2*pcs.shape[0]), (1,)).item()
        sample_indices = torch.randperm(pcs.shape[0])[:n_cells]
        rep_cells = pcs[sample_indices]

        cc_values = a.obs['cc'].values[sample_indices]
        ordered_indices = np.argsort(cc_values)
        ordered_indices_torch = torch.LongTensor(ordered_indices)
        ordered_rep_cells = rep_cells[ordered_indices_torch]

        ordered_rep_cells = torch.cat([ordered_rep_cells, ordered_rep_cells[0].unsqueeze(0)], dim=0)
        dist_vecs = ordered_rep_cells[1:] - ordered_rep_cells[:-1]
        norms = dist_vecs.norm(dim=1)
        topo_loss = norms.sum()
        topo_loss_total += topo_loss

    topo_loss_total.backward()
    grads = pts.grad

    scores = grads.norm(dim=0).numpy()
    sorted_indices = np.argsort(scores)[::-1]

    return scores, grads