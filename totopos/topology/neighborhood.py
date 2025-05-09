"""
Estimate persistent homology lifetimes noise floor. 
"""
import numpy as np 
from tqdm import tqdm
from ripser import ripser
from sklearn.cluster import MiniBatchKMeans as kmeans
from typing import Tuple

def get_lifetimes(dgm):
    death_radius, birth_radius = dgm[:, 1], dgm[:, 0]
    return death_radius - birth_radius
    
def get_largest_lifetime_from_diagram(dgm): 
    lifetimes = get_lifetimes(dgm)
    return sorted(lifetimes, reverse=True)[0]

def largest_neighborhood_lifetime(data, n_clusters=20,ph_dim=1,use_tqdm=True)->Tuple[float, list, np.ndarray]:
    """
    Returns the PH noise floor, lifetimes and labels of neighborhoods.
    The noise floor $\alpha$ is defined as the largest PH lifetime across k neighborhoods using kmeans clustering.

    Params
    ------
    data (np.ndarray)
        Point cloud on which to compute the persistent homology. 

    n_clusters(int)
        Parameter k for Kmeans clustering. 
    
    ph_dim (int)
        Dimension for PH to focus on.
    
    use_tqdm (bool)
        Whether to use tqdm for progress bar. Default is True.
    
    Returns
    -------
    largest_nbd_lifetime, neighborhood_lifetimes, labels
    """
    
    km = kmeans(n_clusters=n_clusters,random_state=13)
    km.fit(data)
    labels=km.predict(data)
    largest_nbd_lifetime=0
    neighborhood_lifetimes = []
    if use_tqdm:
        iterable = tqdm(range(n_clusters),desc="Estimating PH noise floor using Voronoi neighborhoods...")
    else: 
        iterable = range(n_clusters)
    
    for i in iterable:
        persistence_diagram = ripser(data[labels==i],maxdim=ph_dim)["dgms"][ph_dim]
        neighborhood_lifetimes.extend(get_lifetimes(persistence_diagram))
        if len(persistence_diagram)>0:
            largest_lt_clus = get_largest_lifetime_from_diagram(persistence_diagram)
            if largest_lt_clus > largest_nbd_lifetime:
                largest_nbd_lifetime = largest_lt_clus
    return largest_nbd_lifetime, neighborhood_lifetimes, labels


def neighborhood_subsample(data, n_clusters): 
    """
    Returns clustering labels, representative indices and kmeans centroids
    """
    km = kmeans(n_clusters=n_clusters, random_state=13)
    km.fit(data)
    labels = km.predict(data)
    
    # Find the indices of the points closest to the cluster centers
    representative_indices = np.zeros(n_clusters,dtype=int)
    for i,center in enumerate(km.cluster_centers_):
        distances = np.linalg.norm(data - center, axis=1)
        representative_indices[i]= np.argmin(distances)
    
    return labels, representative_indices, km.cluster_centers_