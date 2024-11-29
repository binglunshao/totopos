"""Designed for inferring prominent topology from data.
"""
import numpy as np
from ripser import ripser
import warnings
from .neighborhood import get_lifetimes, get_largest_lifetime_from_diagram, largest_neighborhood_lifetime

class SimplicialTopology(): 
    def __init__(self, data, n_pts = None, maxdim=1):
        self.data = data
        self.n_pts = n_pts 
        self.maxdim = maxdim
    
    def compute_persistent_cohomology(self, mode, verbose=False):
        """Computes Persistent Cohomology using Bauer's ripser algorithm."""
        self.ph = ripser(self.data, n_perm = self.n_pts, maxdim=self.maxdim, prime=2)

    def estimate_neighborhood_threshold(self, ph_dim = 1, neighborhood_size = 300, verbose = False): 
        """Estimates the homology lifetime "noise floor" defined as the largest PH lifetime 
        across k neighborhoods using kmeans clustering."""

        assert ph_dim >= len(self.ph.dgms_)-1, "Desired homology dimension was not computed, please reinstaintate object."
        dgm = self.ph.dgms_[ph_dim]
        lifetimes = get_lifetimes(dgm)
        largest_nbd_lifetime, neighborhood_lifetimes = largest_neighborhood_lifetime(
            self.data, n_clusters=len(self.data)//neighborhood_size, ph_dim = ph_dim
        )
        largest_nbd_lifetime *= 1.1 
        self.neighborhood_lifetime_threshold = {ph_dim:largest_nbd_lifetime}
        n_prominent_feats_ = sum(lifetimes >= largest_nbd_lifetime)
        self.n_prominent_feats = {ph_dim: n_prominent_feats_}
        if verbose: print(f"Found {n_prominent_feats_} prominent topological loops.")

    def fit(self,verbose=False):
        self.compute_persistent_cohomology(verbose=verbose)
        self.estimate_neighborhood_threshold(verbose=verbose)
    
    # def fit_transform(self):
    #     self.fit()
    #     return self.toroidal_coords
    
    # def transform(self): 
    #     return self.toroidal_coords