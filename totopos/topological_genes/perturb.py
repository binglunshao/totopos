import torch
import oineus as oin
import numpy as np 

def get_topological_gene_scores_via_perturbation(
    data:np.ndarray, max_radius:float=2.0, n_threads:int=2, max_hom_dim:int=1, 
    n_topo_feats:int=1,
    ):
    """
    Returns gene scores via a perturbation approach. 
    """

    pts = torch.Tensor(data)
    pts.requires_grad_(True)

    # computer pairwise distances differentiably
    pts1 = pts.unsqueeze(1)
    pts2 = pts.unsqueeze(0)

    epsilon = 1e-8
    sq_dists = torch.sum((pts1 - pts2) ** 2, dim=2)
    dists = torch.sqrt(sq_dists + epsilon)

    # make filtration
    fil = oin.diff.vietoris_rips_pwdists(
        dists, max_dim=max_hom_dim, max_radius=max_radius, n_threads=n_threads
    )
    top_opt = oin.diff.TopologyOptimizer(fil)

    #print(f"{len(fil) = }")
    dgm = top_opt.compute_diagram(include_inf_points=False)
    eps = top_opt.get_nth_persistence(max_hom_dim, n_topo_feats)
    indices, values = top_opt.simplify(eps, oin.DenoiseStrategy.BirthBirth, max_hom_dim)
    critical_sets = top_opt.singletons(indices, values)
    crit_indices, crit_values = top_opt.combine_loss(critical_sets, oin.ConflictStrategy.Max)
    crit_indices = np.array(crit_indices, dtype=np.int32)
    crit_values = torch.Tensor(crit_values)
    top_loss = torch.mean((fil.values[crit_indices] - crit_values) ** 2)
    top_loss.backward()

    return pts.grad.abs().sum(0).numpy()