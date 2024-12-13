import dionysus
import numpy as np 

def reduced_column_to_simplex_ids(col:dionysus.Chain)-> list: 
    """Returns simplices from a chain coming from a column in the reduced matrix R, from an D=RV persistent homology decomposition.
    """
    simplices = str(col).split("+")
    if len(simplices[0])==0:
        return None
    simplices = [spx.strip().split("*")[-1] for spx in simplices]
    return simplices

def dionysus_simplex_to_id_filtvalue(dio_spx):
    "Returns a dionysus simplex in str format and its value in the filtration."
    spx, val = str(dio_spx).split()
    return spx, float(val)
    
def get_pairs(R:dionysus.ReducedMatrix,filt:dionysus.Filtration, maxdim=1)->list:
    """
    Returns persistent pairs given R from a D=RV be a persistent homology decomposition 
    of the filtration boundary matrix D.
    """
    pairs = [list() for i in range(maxdim+1)]
    
    for j, R_j in enumerate(R): 
        simplex_ids = reduced_column_to_simplex_ids(R_j)
        if simplex_ids is not None: 
            
            birth_idx, death_idx = int(simplex_ids[-1]), j
            point_ids_birth, filt_value_birth = dionysus_simplex_to_id_filtvalue(filt[birth_idx])
            point_ids_death, filt_value_death = dionysus_simplex_to_id_filtvalue(filt[death_idx])
            pers = filt_value_death - filt_value_birth
            if pers > 0: 
                dim=len(point_ids_birth.split(","))-1
                pairs[dim].append(((birth_idx, death_idx), pers))
    return pairs

def str_simplex_to_numpy_array(simplex_list)->np.ndarray:
    # Strip '<' and '>' and split each entry by ',' to convert into integers
    int_list = [tuple(map(int, item.strip('<>').split(','))) for item in simplex_list]
    return np.array(int_list)
    
def get_homology_rep_from_persistence_pair(R:dionysus.ReducedMatrix, filt:dionysus.Filtration, pair:tuple)->np.ndarray:
    """
    Returns a persistent homology representative given a reduced matrix R (from an D=RV decomposition),
    its corresponding filtration object, and a persistent pair.
    """
    death_spx_idx = pair[1] # the second entry contains the death simplex
    R_j = R[death_spx_idx] # extract column for death simplex 
    homology_chain_ids = reduced_column_to_simplex_ids(R_j)
    homology_rep = [dionysus_simplex_to_id_filtvalue(filt[int(index)])[0] for index in homology_chain_ids]

    return str_simplex_to_numpy_array(homology_rep)

def homology_generator_dionysus(data:np.ndarray): 
    """
    Returns edges of H_1 persistent homology representative with largest lifetime.
    
    Params
    ------
    data (np.ndarray)
        Input dataset. 

    Returns 
    -------
    homology_representative (list)
        List containing edges of the homology representative. 
    """
    prime = 2  # Use a prime base for the homology coefficients
    f = dionysus.fill_rips(data, k=2, r=40)  # 2-D simplicial complex up to radius 2
    R = dionysus.homology_persistence(f, prime=2)
    dgms = dionysus.init_diagrams(R, f)
    pairs = get_pairs(R,f)
    h1_pairs = pairs[1]
    h1_pairs = sorted(h1_pairs, key = lambda x: x[1])
    homology_representative = get_homology_rep_from_persistence_pair(R, f, h1_pairs[-1][0])
    return homology_representative