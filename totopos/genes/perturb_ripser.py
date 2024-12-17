import torch 
from ripser import ripser 
import numpy as np 
from ..topology.neighborhood import neighborhood_subsample
from ..utils.ph_utils import min_enclosing_radius_torch, get_lifetimes
from ..utils.utils import randomized_pca_torch, differentiable_distance_matrix_torch


def topological_scores_perturbation_torch_ripser(adata, ph=None, n_pcs = 20, max_distance = None): 
    
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

    cocycles=ph["cocycles"]
    dgms=ph["dgms"]
    lifetimes = get_lifetimes(dgms[1])
    ix_largest = np.argsort(lifetimes)[-1]
    cocycle_edges_largest_hom_class = cocycles[1][ix_largest][:, :2] # first two entries are edges 
    cocycle_edges_largest_hom_class = cocycle_edges_largest_hom_class[:, ::-1] # get edges in lexicographic order 
    death_time = dgms[1][ix_largest][1]

    crit_edges_idx_x, crit_edges_idx_y = cocycle_edges_largest_hom_class.T

    filt_values = torch.sum((pcs[crit_edges_idx_x, :] - pcs[crit_edges_idx_y, :])**2, axis=1) # distance of largest edges in critical simplices
    target_crit_values = torch.repeat_interleave(torch.Tensor([death_time]), repeats=len(cocycle_edges_largest_hom_class))
    topo_loss = torch.norm(target_crit_values - filt_values)
    topo_loss.backward()

    gradient = pts.grad
    scores = gradient.norm(dim=0).numpy()

    return scores