"""Designed for computing pseudotime using Circular Coordinates algorithm.
"""
from ..utils.ph_utils import get_lifetimes
import numpy as np
from dreimac import ToroidalCoords
import warnings
import numpy as np
from ripser import ripser
from scipy import sparse
from scipy.sparse.linalg import lsqr

def get_longest_persistent_class_index(dgm_H1):
    """Return index of the longest-living class in H¹."""
    return max(enumerate(dgm_H1), key=lambda x: x[1][1] - x[1][0])[0]

def build_coboundary_matrix_delta_0(distance_matrix, epsilon):
    """Construct coboundary matrix δ⁰ based on distance threshold ε."""
    edge_indices = np.array((distance_matrix <= epsilon).nonzero()).T
    n_edges = edge_indices.shape[0]
    row_idx = np.repeat(np.arange(n_edges), 2)
    col_idx = edge_indices.flatten()
    values = np.tile([-1, 1], n_edges)
    delta_0 = sparse.coo_matrix((values, (row_idx, col_idx)), shape=(n_edges, distance_matrix.shape[0]))
    return delta_0, edge_indices

def convert_cocycle_to_integer_coefficients(alpha_p, prime):
    """Change the coefficients of cocycle over F_p (finite field) to an integer-valued cocycle alpha.
    
    Note: The idea is to use the smallest integer representative for each coefficient (in absolute magnitude).
    """
    alpha = alpha_p[:, 2].copy() #this retrieves the values of the coefficient at each edge 
    alpha[alpha > (prime - 1) / 2] -= prime
    return alpha


def build_edge_cocycle_vector(alpha, cocycle_edges, edges, n_vertices):
    """
    *Extends* the cohomology class alpha, evaluated only on representative cocycle edges, 
    to the whole 1-skeleton (encoded in edges) of the target Vietoris-Rips complex.

    This function evaluates the cocycle `alpha` on the list of edges used to
    build the coboundary matrix δ⁰. It returns a real-valued vector z 
    whose k-th entry is (i_k, j_k), the cocycle value on edge (i_k, j_k),
    with appropriate sign depending on orientation.

    Parameters
    ----------
    alpha : array_like, shape (n_cocycle_edges,)
        Integer values of the cocycle lifted from F_p.
    
    cocycle_edges : array_like, shape (n_cocycle_edges, 2)
        Edge indices (i, j) corresponding to the entries of `alpha`.
        These come from persistent cohomology output and are oriented.
    
    edges : array_like, shape (n_edges, 2)
        Full list of edges (i, j) used to construct δ⁰.
        These are typically all edges with length ≤ ε.

    Returns
    -------
    z : np.ndarray, shape (n_edges,)
        The edge cocycle vector, i.e., z_k = alpha(i_k, j_k),
        with proper sign depending on orientation.
    """
    # Build an antisymmetric representation of the cocycle
    # such that C[i,j] = α(i,j), C[j,i] = -α(i,j)
    C = np.zeros((n_vertices, n_vertices), dtype=np.int32)

    for (i, j), val in zip(cocycle_edges, alpha):
        C[i, j] = val
        C[j, i] = -val  # enforce antisymmetry: α(j,i) = -α(i,j)

    # Evaluate α on each edge in the Rips complex
    # For each edge (i, j), extract C[i, j]
    z = C[edges[:, 0], edges[:, 1]]

    return z

def compute_harmonic_representative(delta_0, z):
    """
    Solve least squares ∥z-𝛿_0(γ)∥^2 to obtain vertex values of the circular coordinates θ = γ mod 1.
    """
    gamma = lsqr(delta_0, z)[0]  # γ : X → ℝ such that alpha_bar = alpha + 𝛿_0 γ 
    theta = np.mod(gamma, 1.0)   # project to S¹ = ℝ / ℤ
    return theta

def compute_circular_coordinate(X, ph=None, prime=47, ix_cohom_class=1):
    """
    Compute a circular coordinate θ : X → S¹ from input point cloud X.

    Steps:
    1. Compute persistent cohomology to get alpha_p over F_p
    2. Convert to integer coefficients alpha_p → alpha ∈ C¹(X; ℤ)
    3. Construct δ⁰ and form edge cocycle vector z = alpha(edges)
    4. Minimize ||z - δ⁰ f||² → θ = f mod 1

    Parameters
    ----------
    X : array_like
        Input point cloud data (n_samples, n_features).
    ph : dict, optional
        Precomputed persistent cohomology output from ripser(do_cocycles=True,coeff=prime)
        If None, it will be computed.
    prime : int, optional
        Prime number for field coefficients in persistent cohomology (default: 47).
    ix_cohom_class : int, optional
        Index (from most to least persistent) of the cohomology class to use. 
        Defaults to most persistent class

    Returns
    -------
    theta : np.ndarray
        Circular coordinate values for each point in X, in [0, 1).

    Notes
    -----
    Paper: https://arxiv.org/abs/0905.4887
    Based on https://github.com/appliedtopology/circular-coordinates/
    """
    if ph is None:
        ph = ripser(X, coeff=prime, do_cocycles=True)
    
    dgm_H1 = ph['dgms'][1]
    cocycles_H1 = ph['cocycles'][1]
    D = ph['dperm2all']
    lifetimes = get_lifetimes(dgm_H1)

    # if ix_cohomology_class is None:
    #     ix_cohomology_class = get_longest_persistent_class_index(dgm_H1)
    
    ix_cohomology_class = np.argsort(lifetimes)[-ix_cohom_class]
    
    epsilon = dgm_H1[ix_cohomology_class][1]

    delta_0, edges = build_coboundary_matrix_delta_0(D, epsilon)
    alpha_p = cocycles_H1[ix_cohomology_class]
    alpha = convert_cocycle_to_integer_coefficients(alpha_p, prime)

    # extend the cohomology class to all edges in VR complex
    z = build_edge_cocycle_vector(alpha, alpha_p[:, :2], edges, D.shape[0])
    theta = compute_harmonic_representative(delta_0, z)
    return theta


class CyclicTopologyPseudotime():
    """DEPRECATED
    Class to compute pseudotime coordinates for cyclic trajectories.
    """ 
    def __init__(self, data, n_pts = None):
        """
        Parameters
        ----------
        data (np.array)

        n_pts (int)
            Subsample size to compute homology and pseudotime coordinates. 
        """
        self.data = data
        self.n_pts = n_pts if n_pts is not None else len(data)
    
    def compute_persistent_cohomology(self, verbose=False):
        """Computes Persistent Cohomology using Bauer's ripser algorithm."""
        self.ph = ToroidalCoords(
            self.data, n_landmarks=self.n_pts, maxdim=1, prime=47, verbose=verbose
        )

    def fit(self, perc = .1, check_consistency = False, n_classes = None):
        """Computes harmonic representatives using Scoccola et al. Toroidal Coordinates algorithm.
        """
        #assert 1 in self.n_prominent_feats.keys(), f"Please run estimate_neighborhood_threshold() for dim 1."
        n_classes = self.n_prominent_feats[1] if n_classes is None else n_classes
        self.toroidal_coords = self.ph.get_coordinates(
            perc=perc, 
            cocycle_idxs = range(n_classes), 
            standard_range=False, 
            check_cocycle_condition=True
        ).T

        if check_consistency: 
            self.toroidal_coords_consistency_check()

    # def fit(self,verbose=False):
    #     self.compute_persistent_cohomology(verbose=verbose)
    #     self.estimate_neighborhood_threshold(verbose=verbose)
    #     self.get_harmonic_reps()
    
    def fit_transform(self):
        self.fit()
        return self.toroidal_coords
    
    def transform(self): 
        return self.toroidal_coords

    def get_harmonic_rep_indicators(self): 
        return self.toroidal_coords > np.pi

    def is_invalid_harmonic_rep(self, index, thresh = .5): 
        fraction_nz_pts = sum(self.toroidal_coords[:, index] > np.pi)/len(self.data)
        return fraction_nz_pts > thresh

    def correct_for_one_inconsistent_toroidal_coord(self, coords, invalid_idx):
        indicators = self.get_harmonic_rep_indicators()
        indicators[:, invalid_idx] = indicators.sum(1) % 2
        self.toroidal_coords[:, invalid_idx] = self.toroidal_coords[:, invalid_idx][indicators[:, invalid_idx]]
        return coords

    def consistency_check(self):
        invalid = [
            self.is_invalid_harmonic_rep(self.toroidal_coords[:, i]) for i in range(self.n_prominent_feats)
        ]

        if np.sum(invalid)>0:
            if np.sum(invalid)==1:
                invalid_idx = np.flatnonzero(invalid)[0]
                self.toroidal_coords = self.correct_for_one_inconsistent_toroidal_coord(self.toroidal_coords, invalid_idx)
            else:
                warnings.warn(f"There are {np.sum(invalid)}/{self.n_prominent_feats} invalid harmonic reps, could not correct them.")