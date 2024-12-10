"""
Estimate persistent homology lifetimes noise floor. 
"""
import numpy as np 
from tqdm import tqdm
from ripser import ripser
from sklearn.cluster import MiniBatchKMeans as kmeans

def get_lifetimes(dgm):
    death_radius, birth_radius = dgm[:, 1], dgm[:, 0]
    return death_radius - birth_radius
    
def get_largest_lifetime_from_diagram(dgm): 
    lifetimes = get_lifetimes(dgm)
    return sorted(lifetimes, reverse=True)[0]

def largest_neighborhood_lifetime(
    data, 
    n_clusters=20,
    ph_dim=1
): 
    """Returns the noise floor $\alpha$ defined as the largest PH lifetime across k neighborhoods using kmeans clustering.

    Params
    ------
    data (np.ndarray)
        Point cloud on which to compute the persistent homology. 

    n_clusters(int)
        Parameter k for Kmeans clustering. 
    
    ph_dim (int)
        Dimension for PH to focus on.
    """
    
    km = kmeans(n_clusters=n_clusters)
    km.fit(data)
    labels=km.predict(data)
    largest_nbd_lifetime=0
    neighborhood_lifetimes = []
    for i in tqdm(range(n_clusters),desc="Estimating PH noise floor using Voronoi neighborhoods..."):
        persistence_diagram = ripser(data[labels==i],maxdim=ph_dim)["dgms"][ph_dim]
        neighborhood_lifetimes.extend(get_lifetimes(persistence_diagram))
        if len(persistence_diagram)>0:
            largest_lt_clus = get_largest_lifetime_from_diagram(persistence_diagram)
            if largest_lt_clus > largest_nbd_lifetime:
                largest_nbd_lifetime = largest_lt_clus
    return largest_nbd_lifetime, neighborhood_lifetimes


def neighborhood_subsample(data, n_clusters): 
    """
    Returns clustering labels, representative indices and kmeans centroids
    """
    km = kmeans(n_clusters=n_clusters)
    km.fit(data)
    labels=km.predict(data)

    _, representative_indices = np.unique(labels, return_index=True)
    
    return labels, representative_indices, km.cluster_centers_