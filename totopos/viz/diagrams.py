import numpy as np
from matplotlib import pyplot as plt
from .palettes import *
import copy

def plot_pers_diag_ripser(dgms:list, ax = None, dot_size = 40, conf_int=None):
    """
    Plot persistence diagrams using custom color palette.

    Params
    ------
    dgms (list of np.ndarrays)
        Coordinates for (births, deaths) of each persistent feature across dimensions.
        The i-th list is the persistent diagram of dimension i.
    """
    n_dgms = len(dgms)
    dgms_, max_val = replace_inf(dgms)
    ax = ax or plt.gca()
    pal = caltech_palette()
    
    ax.scatter(
        *dgms_[0].T,
        linewidth=0.1,
        alpha=0.7,
        s = dot_size,
        color=pal[1],
        edgecolors="lightgrey",
        label="$H_0$"
    )

    for i in range(1, n_dgms):
        
        ax.scatter(
            *dgms[i].T,
            linewidth=0.1,
            alpha=0.7,
            s = dot_size,
            color=pal[i+1],
            edgecolors="lightgrey",
            label= f"$H_{i}$"
        )

    ax.axhline(max_val, linestyle="--", color="grey", label="$\infty$")
    ax.plot([0,max_val+.5], [0, max_val+.5], color = "grey")

    if conf_int is not None: 
        ax.fill_between(x= [0, max_val+.5] , y1= [0, max_val+.5], y2=[conf_int,  max_val + conf_int + .5], alpha = 0.3, color = "lightgrey")

    plt.xlim(-.1, max_val)
    plt.ylim(0, max_val +.5)
    plt.xlabel("birth")
    plt.ylabel("death")

    plt.legend(bbox_to_anchor = (1,1))


def replace_inf(arrays):
    """Given a list of persistence diagrams (birth,death pairs) returns diagrams by modifying 
    death values set to infty to the largest finite death time across all diagrams, and the largest death time.

    Params
    ------
    arrays (list)
        List of (n,2) persistence diagrams.
    
    Returns
    -------
    modified_arrays (list)
        List of modified persistence diagrams.
    
    max_val (float)
        Death time with largest (finite) magnitude.
    """
    max_val = -np.inf
    for array in arrays:
        max_val = max(max_val, np.max(array[np.isfinite(array[:,1]), 1]))
    
    max_val += .3 # add an extra quantity for visualization purposes

    modified_arrays = []
    for array in arrays:
        if np.any(np.isinf(array[:, 1])):
            mod_array = copy.deepcopy(array)
            mod_array[mod_array[:,1] == np.inf, 1] = max_val
            modified_arrays.append(mod_array)
        else: 
            modified_arrays.append(array)

    return modified_arrays, max_val