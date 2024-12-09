import numpy as np 

def min_enclosing_radius_torch(D):
    return D.max(1).values.min()

def min_enclosing_radius_subset_torch(D, subset_size=1000):
    if D.shape[0]>subset_size:
        randixs=np.random.choice(D.shape[0], size=subset_size,replace=False)
        return D[randixs].max(1).values.min()
    else:
        return D.max(1).values.min()

def get_lifetimes(dgm):
    """Returns lifetimes given a persistence diagram with format (n,2) array of birth-death pairs"""
    return dgm[:, 1]- dgm[:, 0]