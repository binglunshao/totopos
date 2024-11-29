import torch
import oineus as oin
import numpy as np 
from typing import Tuple

def topological_gene_scores_via_simplification(
    data:np.ndarray, n_threads:int=2, hom_dim:int=1, n_topo_feats:int=1, verbose:bool = False, pca:bool = False, n_pcs:int=30
    )->np.ndarray:
    """
    Returns gene scores via topological simplification, i.e. reducing the topological noise from a persistent diagram.
    """
    pts = torch.Tensor(data)
    pts.requires_grad_(True)

    if pca:
        if verbose:print("Calculating SVD...") 
        pts_ = pts - pts.mean(dim=0) # mean center data
        U, s, Vt = torch.svd_lowrank(pts_, q = n_pcs + 50, niter = 2)
        pcs = U[:, :n_pcs] *  s[:n_pcs]
        pts1 = pcs.unsqueeze(1)
        pts2 = pcs.unsqueeze(0)
        if verbose:print("Finished SVD calculation.") 
    else:
        pts1 = pts.unsqueeze(1)
        pts2 = pts.unsqueeze(0)

    epsilon = 1e-8
    if verbose:print("Calculating distances...") 
    sq_dists = torch.sum((pts1 - pts2) ** 2, dim=2)
    dists = torch.sqrt(sq_dists + epsilon)
    max_dist = dists.max()
    if verbose:print("Finished differentiable distance calculation.") 

    if verbose:print("Calculating Vietoris-Rips filtration...")
    vr_filtration = oin.diff.vietoris_rips_pwdists(
        dists, 
        max_dim=hom_dim+1, #need the k+1 skeleton for k-homology
        max_radius=max_dist/2, 
        n_threads=n_threads
    )
    if verbose:print(f"Finished filtration {vr_filtration}.")

    top_opt = oin.diff.TopologyOptimizer(vr_filtration)
    
    if verbose:print("Computing persistent homology...")
    dgm = top_opt.compute_diagram(include_inf_points=False)
    if verbose:print("Finished persistent homology calculation.")
    eps = top_opt.get_nth_persistence(hom_dim, n_topo_feats)
    
    if verbose:print("Calculating gene scores...")
    indices, values = top_opt.simplify(eps, oin.DenoiseStrategy.BirthBirth, hom_dim)
    critical_sets = top_opt.singletons(indices, values)
    crit_indices, crit_values = top_opt.combine_loss(critical_sets, oin.ConflictStrategy.Max)
    crit_indices = np.array(crit_indices, dtype=np.int32)
    crit_values = torch.Tensor(crit_values)
    #top_loss = torch.mean((vr_filtration.values[crit_indices] - crit_values) ** 2)
    top_loss = torch.norm(vr_filtration.values[crit_indices] - crit_values)
    top_loss.backward()
    if verbose:print("Finished gene score calculation succesfully.")
    gradient = pts.grad

    return torch.norm(gradient, dim = 0).numpy(), [dgm[i] for i in range(hom_dim+1)]

def topology_layer_perturbation(
    pts:torch.Tensor, hom_dim:int=1, n_threads:int=16, pca:bool=False, n_pcs:int=20, verbose:bool=False
)->Tuple[torch.tensor, list]:
    """
    Returns topological loss and persistent diagrams for perturbation approach.
    TODO: Handle `n_top_feats`

    Params
    ------
    pts : Input dataset. Needs to have the `requires_grad` param active.

    Usage
    -----
    lr = 1e-2
    topo_loss, dgms = topology_layer_perturbation(pts)
    topo_loss.backward()
    grad = pts.grad 
    pts = pts - lr * grad # perturb points 

    TODO: add possibility of calculating via subset. 
    """
    if pca:
        pts_ = pts - pts.mean(dim=0) # mean center data
        U, s, Vt = torch.svd_lowrank(pts_, q = n_pcs + 50, niter = 2)
        pcs = U[:, :n_pcs] *  s[:n_pcs]
        pts1 = pcs.unsqueeze(1)
        pts2 = pcs.unsqueeze(0)
    else:
        pts1 = pts.unsqueeze(1)
        pts2 = pts.unsqueeze(0)

    epsilon = 1e-8
    sq_dists = torch.sum((pts1 - pts2) ** 2, dim=2)
    dists = torch.sqrt(sq_dists + epsilon)
    max_dist = dists.max()
    dists_np = dists.detach().numpy().astype(np.float64)
    if verbose:print("Calculating Vietoris-Rips filtration...")
    fil, longest_edges = oin.get_vr_filtration_and_critical_edges_from_pwdists(
        dists_np, max_dim=2, max_radius = max_dist/2 + .1, n_threads=n_threads
    )
    if verbose:print(f"Finished filtration {fil}.")

    if verbose:print("Computing persistent homology...")
    dualize = False # no cohomology
    dcmp = oin.Decomposition(fil, dualize) # create VRU decomposition object, does not perform reduction yet

    rp = oin.ReductionParams()
    rp.compute_u = False # U matrix cannot be computed in parallel mode
    rp.compute_v = True
    rp.n_threads = 16
    dcmp.reduce(rp) # perform PH reduction
    if verbose:print("Finished persistent homology calculation.")

    if verbose:print("Calculating gene scores...")
    dgms = dcmp.diagram(fil, include_inf_points=False)
    dgms = [dgms[0], dgms[1]]
    lifetimes = dgms[1][:, 1] - dgms[1][:, 0]
    ix_largest_lifetime = np.argsort(lifetimes)[-1]

    indices_dgm = dcmp.diagram(fil).index_diagram_in_dimension(hom_dim)
    birth_simplex_sorted_ix, death_simplex_sorted_ix = indices_dgm[ix_largest_lifetime, :]
    
    tgt_val = dgms[hom_dim][ix_largest_lifetime].sum()/2 # send ph class to diag
    indices = [birth_simplex_sorted_ix, death_simplex_sorted_ix]
    values = [tgt_val, tgt_val]

    opt = oin.TopologyOptimizer(fil)
    critical_sets = opt.singletons(indices, values)
    crit_indices, target_crit_values = opt.combine_loss(critical_sets, oin.ConflictStrategy.Max)
    crit_edges = longest_edges[crit_indices, :]
    crit_edges_x, crit_edges_y = crit_edges[:, 0], crit_edges[:, 1]
    target_crit_values = torch.Tensor(target_crit_values)
    init_crit_values = torch.sum((pts[crit_edges_x, :] - pts[crit_edges_y, :])**2, axis=1) # distance of largest edges in critical simplices
    topo_loss = torch.norm(target_crit_values - init_crit_values)
    if verbose:print("Finished gene score calculation succesfully.")

    return topo_loss, [dgms[i] for i in range(hom_dim+1)]

def topological_gene_scores_via_perturbation(
    data:np.ndarray, n_threads:int=2, hom_dim:int=1, n_topo_feats:int=1, verbose:bool = False, epochs:int= 1, pca:bool=False, n_pcs:int=20
)->Tuple[np.ndarray,list]:
    """
    Returns gene scores and persistent diagrams via a perturbation approach. 
    In particular, this method calculates the norm of the gradients w.r.t. 
    a perturbation of the input points to send the largest persistent homology class to the diagonal.
    """

    pts = torch.Tensor(data)
    pts.requires_grad_(True)
    topo_loss, dgms = topology_layer_perturbation(pts, hom_dim, n_threads, pca, n_pcs, verbose)
    topo_loss.backward()
    grad = pts.grad
    return grad.norm(dim=0).numpy(), [dgms[i] for i in range(hom_dim+1)]