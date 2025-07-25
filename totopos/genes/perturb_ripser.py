import torch 
from ripser import ripser 
import numpy as np 
from ..topology.neighborhood import neighborhood_subsample
from ..utils.ph_utils import min_enclosing_radius_torch, get_lifetimes
from ..utils.utils import randomized_pca_torch
import anndata as ad

def topological_scores_perturbation_torch_ripser(
    adata: ad.AnnData,
    ph: dict = None, 
    n_pcs: int = 20, 
    max_distance: float = None, 
    ix_top_class: int = 1
) -> tuple:
    """
    Computes topological scores and gradients for perturbations using Ripser.

    Params
    ------
    - adata (AnnData): Annotated data matrix.
    - ph (dict, optional): Precomputed persistent homology. Defaults to None.
    - n_pcs (int, optional): Number of principal components to use. Defaults to 20.
    - max_distance (float, optional): Maximum distance threshold for persistence computation. Defaults to None.
    - ix_top_class (int, optional): Index of the top homology class to consider. Defaults to 1.

    Returns
    -------
    - tuple: Gradients (torch.Tensor) and topological ranking scores (numpy.ndarray).
    """
    data = adata.X.A
    pts = torch.Tensor(data)
    pts.requires_grad_(True);
    pcs = randomized_pca_torch(pts, n_pcs)
    pcs_np = pcs.detach().numpy()

    if ph==None:
        ph = ripser(
            pcs_np,
            do_cocycles=True, 
            thresh=np.inf if max_distance is None else max_distance*1.1
        )

    topological_ranking_scores, gradients = compute_topological_scores_perturbation(data, pcs, ph, ix_top_class)

    return topological_ranking_scores, gradients


def compute_topological_scores_perturbation(data, pcs, ph, ix_top_class: int = 1):
    """
    Compute topological ranking scores based on persistent omology.
    Identifies the most persistent cohomology class (by default, the top class) from the persistence diagram,
    extracts its representative cocycle edges, and computes a topological loss based on the squared distances
    between critical points in the principal component space. The loss is backpropagated to obtain gradients,
    which are then used to rank features by their topological relevance.
    Params
    ------ 
    ix_top_class (int, optional): Index (from the top) of the persistent cohomology class to analyze. 
        Defaults to 1 (the most persistent class).
    Returns
    -------
    Tuple[np.ndarray, torch.Tensor]: 
        - topological_ranking_scores: Feature-wise ranking scores derived from gradient norms.
        - gradients: Raw gradients of the data with respect to the topological loss.
    """

    cocycles, dgms = ph["cocycles"], ph["dgms"]

    
    lifetimes = get_lifetimes(dgms[1])
    ix_cohom_class = np.argsort(lifetimes)[-ix_top_class]
    cocycle_edges_largest_hom_class = cocycles[1][ix_cohom_class][:, :2] # first two entries are edges 
    cocycle_edges_largest_hom_class = cocycle_edges_largest_hom_class[:, ::-1] # get edges in lexicographic order 
    death_time = dgms[1][ix_cohom_class][1]

    crit_edges_idx_x, crit_edges_idx_y = cocycle_edges_largest_hom_class.T

    filt_values = torch.sum((pcs[crit_edges_idx_x, :] - pcs[crit_edges_idx_y, :])**2, axis=1) # distance of largest edges in critical simplices
    target_crit_values = torch.repeat_interleave(torch.Tensor([death_time]), repeats=len(cocycle_edges_largest_hom_class))
    topo_loss = torch.norm(target_crit_values - filt_values)
    topo_loss.backward(retain_graph=True) # to run multiple times, works since grads are computed w.r.t. to different slices of data

    gradients = data.grad
    topological_ranking_scores = gradients.norm(dim=0).numpy()
    return topological_ranking_scores, gradients