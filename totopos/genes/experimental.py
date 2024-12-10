from ripser import ripser
from typing import Tuple
import numpy as np
import torch 
import oineus as oin
from ..utils.ph_utils import min_enclosing_radius_torch, get_lifetimes
from ..utils.utils import randomized_pca_torch, differentiable_distance_matrix_torch

def topological_gene_scores_via_perturbation(
    data:np.ndarray, n_pts:int=None, n_threads:int=2, hom_dim:int=1, n_topo_feats:int=1, max_distance:float=None,
    verbose:bool = False, pca:bool = False, n_pcs:int=30, target_strategy:str="death-death"
    )->Tuple[list, np.ndarray]:
    """
    Returns gene scores using a modification of the perturbation method
    """
    n_pts = len(data) if n_pts is None else n_pts
    # thresh could be a little more than death time of the tgt hom class 
    ph = ripser(data, n_perm=n_pts, do_cocycles=True, thresh=np.inf if max_distance is None else max_distance*1.1)
    cocycles=ph["cocycles"]
    dgms=ph["dgms"]
    lifetimes = get_lifetimes(dgms[1])
    ix_largest = np.argsort(lifetimes)[-1]
    cocycle_edges_largest_hom_class = cocycles[1][ix_largest][:, :2]
    cocycle_edges_largest_hom_class = cocycle_edges_largest_hom_class[:, ::-1]
    death_time = dgms[1][ix_largest][1]

    pts = torch.Tensor(data)
    pts.requires_grad_(True);
    if pca: pts = randomized_pca_torch(pts, n_pcs)
    dists = differentiable_distance_matrix_torch(pts)

    if verbose:print("Calculating Vietoris-Rips filtration...")
    
    max_distance = 2*min_enclosing_radius_torch(dists) if max_distance is None else max_distance + .2
    
    vr_filtration = oin.diff.vietoris_rips_pwdists(
        dists, 
        max_dim=hom_dim+1, #need the k+1 skeleton for k-homology
        max_radius=death_time*1.5, # max_radius in oineus is really max_distance... 
        n_threads=n_threads
    )

    simplices = [spx.vertices for spx in vr_filtration.cells()]
    crit_indices=[simplices.index(list(spx)) for spx in cocycle_edges_largest_hom_class]
    crit_values = torch.repeat_interleave(torch.Tensor([death_time]), repeats=len(crit_indices))
    top_loss = torch.norm(vr_filtration.values[crit_indices] - crit_values)
    top_loss.backward()
    gradient = pts.grad

    return gradient.norm(dim=0).numpy()