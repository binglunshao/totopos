import numpy as np 

def min_enclosing_radius(D):
    return D.max(1).min()

def min_enclosing_radius_subset(D, subset_size=1000):
    if D.shape[0]>subset_size:
        randixs=np.random.choice(D.shape[0], size=subset_size,replace=False)
        return D[randixs].max(1).min()
    else:
        return D.max(1).min()