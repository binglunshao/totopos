"""Designed for computing pseudotime using Circular Coordinates algorithm.
"""

import numpy as np
from dreimac import ToroidalCoords
# from ripser import ripser
import warnings

class CyclicTopologyPseudotime(): 
    def __init__(self, data, n_pts = None, maxdim=1):
        self.data = data
        self.n_pts = n_pts 
        self.maxdim = maxdim
    
    def compute_persistent_cohomology(self, verbose=False):
        """Computes Persistent Cohomology using Bauer's ripser algorithm."""
        self.ph = ToroidalCoords(
            self.data, n_landmarks=self.n_pts, maxdim=self.maxdim, prime=2, verbose=verbose
        )

    def fit(self, perc = .1, check_consistency = False, n_classes = None):
        """Computes harmonic representatives using Scoccola et al. Toroidal Coordinates algorithm.
        """
        assert 1 in self.n_prominent_feats.keys(), f"Please run estimate_neighborhood_threshold() for dim 1."
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