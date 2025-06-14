import numpy as np
import anndata as ad
from tqdm import tqdm
import time
from ripser import ripser
import networkx as nx
import heapq
from sklearn.neighbors import BallTree

def prim_tree_find_loop(graph: nx.Graph, critical_edge: tuple, points: np.ndarray):
    """
    Returns the MST edges and the cycle formed by adding the critical edge.

    Params
    --------
    graph (nx.Graph)
        A Vietoris-Rips NetworkX graph.
    critical_edge (tuple)
        The critical edge to add to the tree.
    points (np.ndarray)
        The points in the graph.

    Returns
    --------
    (mst_edges, cycle)
        A tuple of the MST edges and the cycle formed by adding the critical edge.
    """
    u_edge, v_edge = critical_edge[0], critical_edge[1]

    for start_vertex in [u_edge, v_edge]:
        if start_vertex not in graph:
            continue
        
        mst_edges = []
        visited = set([start_vertex])
        priority_queue = []

        # Add edges from the start vertex to the priority queue, excluding the critical edge
        for neighbor in graph[start_vertex]:
            if neighbor not in visited:
                # Exclude the critical edge from the MST building phase
                if not ((start_vertex == u_edge and neighbor == v_edge) or (start_vertex == v_edge and neighbor == u_edge)):
                    weight = graph[start_vertex][neighbor]['weight']
                    heapq.heappush(priority_queue, (weight, start_vertex, neighbor))

        # Grow the tree for the remaining vertices
        while priority_queue:
            weight, u, v = heapq.heappop(priority_queue)
            if v not in visited:
                # Add the edge to the tree
                mst_edges.append((u, v, weight))
                visited.add(v)

                # Add edges from the new vertex to the priority queue
                for neighbor in graph[v]:
                    if neighbor not in visited:
                        if not ((v == u_edge and neighbor == v_edge) or (v == v_edge and neighbor == u_edge)):
                            weight = graph[v][neighbor]['weight']
                            heapq.heappush(priority_queue, (weight, v, neighbor))

        # After building the tree, check if adding the critical edge forms a cycle using NetworkX
        # Build the graph using NetworkX
        G = nx.Graph()
        G.add_weighted_edges_from(mst_edges)

        # Check if both nodes of the critical edge are in the tree
        if u_edge in G.nodes and v_edge in G.nodes:
            # Add the critical edge with weight calculated from points
            weight_edge = np.linalg.norm(points[u_edge] - points[v_edge])
            G.add_edge(u_edge, v_edge, weight=weight_edge)
            try:
                # Attempt to find a cycle starting from u_edge
                cycle = nx.find_cycle(G, source=u_edge)
                cycle = np.array(cycle)
                # print(f"Cycle formed after adding critical edge {edge} starting from vertex '{start_vertex}':")
                # print("Cycle:", cycle)
                # Return the MST edges plus the critical edge
                return mst_edges, cycle
            except nx.exception.NetworkXNoCycle:
                # No cycle formed, continue to next edge
                pass
        else:
            # Critical edge nodes not in tree, cannot form cycle
            pass

            # If no cycle is formed, or critical edge nodes are not in tree, backtrack and try next edge
        # If no valid tree is found with the current starting vertex, continue to the next starting vertex

    # If no such tree is found, return None
    # print("No cycle formed with the given starting vertices and critical edge.")
    return None

def vietoris_rips_graph(point_cloud: np.ndarray, birth_distance: float) -> nx.Graph:
    """
    Returns a Vietoris-Rips graph from a point cloud using the birth distance as a threshold.

    Params
    --------
    point_cloud (np.ndarray)
        The point cloud to create the graph from.
    birth_distance (float)
        The birth distance threshold of the cocycle.
    Returns
    --------
    nx.Graph
        The Vietoris-Rips graph.
    """
    birth_distance += 1e-4
    G = nx.Graph()
    num_points = len(point_cloud)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = np.linalg.norm(point_cloud[i] - point_cloud[j])
            if dist <= birth_distance:
                G.add_edge(i, j, weight=dist)
    return G

def get_prominent_cohomology_class_data(ph_output: dict, n: int = 5, method:str = "ripser") -> list:
    """
    Extracts the top n most persistent cohomology generator data from ripser PH output.

    Params
    --------
    ph_output (dict)
        The PH output dictionary containing 'dgms' and 'cocycles'.
    n (int, optional)
        The number of top persistent cocycles to extract, default is 5.
    method (str, default = "ripser")
        One of [`ripser`, `dreimac`]. Specifies the type of PH output.

    Returns
    --------
    cohomology_class_data (list)
        A list of dictionaries containing the birth distance, cocycle, critical edge, and persistence.
    """
    assert method in ["ripser", "dreimac"], f"Method {method} not supported. Choose one of ['ripser', 'dreimac']"

    # Extract the persistence diagram for H1 (1-dimensional features)
    h1_dgm = ph_output['dgms'][1] if method == "ripser" else ph_output.dgms_[1]
    
    # Calculate persistence (death - birth) and sort by persistence in descending order
    persistence = h1_dgm[:, 1] - h1_dgm[:, 0]
    sorted_indices = np.argsort(persistence)[::-1]
    
    # Get the top n most persistent cocycles
    top_n_indices = sorted_indices[:n]
    
    cohomology_class_data = []
    for i in top_n_indices:
        cocycle = ph_output['cocycles'][1][i] if method == "ripser" else ph_output.cocycles_[1][i]
        birth_distance = h1_dgm[i, 0]
        u, v, coeff = cocycle[0]
        critical_edge = (u, v)
        pers = persistence[i]
        cohomology_class_data.append(
            {"birth_dist":birth_distance, "cocycle":cocycle, "critical_edge":critical_edge, "pers": pers}
        )

    return cohomology_class_data

def get_all_loop_nodes(persistent_cohomology_class_data:list, points:np.ndarray) -> list:
    """
    Returns a list of nodes in all loops from the top cocycles.

    Params
    --------
    persistent_cohomology_class_data (list)
        The top cocycles extracted from the ripser result.
    points (np.ndarray)
        The point cloud.

    Returns
    --------
    cycle_nodes (list)
        A list of nodes in all loops.
    """

    birth_distance, _, edge, _ = persistent_cohomology_class_data[0]
    sparse_G = vietoris_rips_graph(points, birth_distance)
    _, cycle = prim_tree_find_loop(sparse_G, edge, points)
    reps = [cycle]

    for i in range(1, len(persistent_cohomology_class_data)):
        birth_distance, _, edge, _ = persistent_cohomology_class_data[i]
        _, cycle = prim_tree_find_loop(sparse_G, edge, points)
        reps.append(cycle)

    cycle_nodes = set()
    for cycle in reps:
        for edge in cycle:
            cycle_nodes.update(edge)

    return list(cycle_nodes)


def get_loop_neighbors(all_data: np.ndarray, query_data: np.ndarray, radius: float, leaf_size: int = 40) -> tuple:
    """
    Returns the neighbors of the loop nodes in the original data using a BallTree.

    Params
    --------
    all_data (np.ndarray)
        The original data.
    query_data (np.ndarray)
        The loop nodes.
    radius (float)
        The radius of the neighborhood.
    leaf_size (int, optional)
        The leaf size of the BallTree, default is 40.

    Returns
    --------
    tuple
        A tuple of the neighbor data and indices.
    """
    tree = BallTree(all_data, leaf_size=leaf_size)
    inds = tree.query_radius(query_data, r=radius)
    unique_inds = np.unique(np.concatenate(inds))
    topocells = all_data[unique_inds]
    return topocells, unique_inds

def critical_edge_method(
    data:np.ndarray, ph:dict=None, npts:int=None, n_loops:int = 1, verbose:bool=False, method:str = "ripser", compute_topocells:bool=True
    )->list: 
    """
    Returns a list homology data for `n_loops` with largest lifetimes in Dgm_1(data). 

    Params
    ------
    data(np.ndarray)
        Input dataset to extract largest cycle from. 
    
    ph (dict, optional)
        Precomputed PH output from ripser. If None, program will compute PH from scratch.

    npts (int, optional)
        Number of points to use for calculation. If None, all points will be used.
    
    n_loops(int)

    verbose(bool, default=False)
        If set to True, print statements of algorithm steps will be delidered. 
    
    method (str, default = "ripser")
        One of [`ripser`, `dreimac`]. Specifies the type of PH output.        

    compute_topocells (bool, default=True)
        If set to True, the function will compute the topoCells for the loop.

    Returns
    -------
    topological_loop_data (list)
        List of `n_loops` dictionaries containing topological information. 
        The item for key "loop" is a size (n,2) numpy array containing the edges of the 
        topological loop with largest lifetime in the PH computation.
    """
    assert method in ["ripser", "dreimac"], f"Method {method} not supported. Choose one of ['ripser', 'dreimac']"

    if ph == None:
        if verbose: print("Starting de novo PH computation...")
        if npts > data.shape[0]:
            npts = None  # Use all points if npts is larger than the number of points in data
        ph = ripser(data, do_cocycles=True, n_perm=npts)
        if verbose:print("Finished computing PH.")

    idx_perm = ph['idx_perm'] if 'idx_perm' in ph else np.arange(data.shape[0])
    points = data[idx_perm]
    full2sub = {full:sub for sub, full in enumerate(idx_perm)}
    sub2full = {sub:full for sub, full in enumerate(idx_perm)}

    topological_loop_data = get_prominent_cohomology_class_data(ph,n=n_loops, method=method)
    birth_dist_latest = topological_loop_data[0]["birth_dist"]
    if verbose:print("Starting VR graph construction.")
    one_skeleton = vietoris_rips_graph(points, birth_dist_latest)
    if verbose:print(f"Finished VR graph. Starting loop discovery...")

    iterable = range(n_loops) if n_loops==1 else tqdm(range(n_loops))
    for i in iterable:
        birth_dist=topological_loop_data[i]["birth_dist"]
        crit_edge=topological_loop_data[i]["critical_edge"]
        crit_edge_sub = (full2sub[crit_edge[0]], full2sub[crit_edge[1]])
        if verbose:print(f"Starting {i+1}-th loop discovery...")
        _, topological_loop = prim_tree_find_loop(one_skeleton, crit_edge_sub, points)
        if verbose:print(f"Finished computing loop {i+1} from VR graph.")
        topological_loop_data[i]["loop"] = topological_loop

        if compute_topocells:
            zero_sk = np.unique(topological_loop)
            zero_sk_full = np.array([sub2full[ix] for ix in zero_sk])

            topocells, tpc_ixs = get_loop_neighbors(
                all_data = data,
                query_data = data[zero_sk_full],
                radius = birth_dist,
            )

            topological_loop_data[i]["topocells"] = topocells
            topological_loop_data[i]["topocell_ixs"] = tpc_ixs
    
    if n_loops==1:
        return [topological_loop_data[0]]

    if verbose: print("Finished critical edge algorithm.")
    return topological_loop_data