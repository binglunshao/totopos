from sklearn.feature_selection import mutual_info_regression as MI
from tqdm import tqdm
import numpy as np

def topological_gene_scores_harmonic_modes(gex_data, Q, idx_ev_start = 0, n_evecs = 10):
    """
    Returns the mutual information (MI) of gene expression vs gene harmonic modes (laplacian eigvenvectors). 
    The function computes MI for the set of eigenvectors Q[:, idx_ev_start : idx_ev_start + n_evecs], 
    i.e. a window of `n_evecs` starting from an index `idx_ev_start`.

    Params
    ------
    gex_data (np.ndarray)
        Gene expression data
    
    Q (np.ndarray)
        Laplacian eigenvectors or other transiently active function over cells
    
    idx_ev_start(int)
        Index of eigenvector to start the computation
    
    n_evecs (int)
        Number of eigenvectors to compute.
    
    Returns
    -------
    mutual_info_scores (np.ndarray)
    """

    n_cells, n_genes = gex_data.shape
    mutual_info_scores=np.zeros((n_genes, n_evecs))

    for j in range(idx_ev_start, idx_ev_start + n_evecs):

        for i in tqdm(range(n_genes), desc = "Computing MI(gene_harmonic_mode,gene)"): 
            mutual_info_scores[i, j-idx_ev_start] = MI(
                X=Q[:, j].reshape(-1,1), 
                y=np.asarray(gex_data[:, i]).flatten()
            )

    return mutual_info_scores