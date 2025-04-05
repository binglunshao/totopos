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
        # Get all edges from the start vertex, sorted by weight
        edges_from_start = sorted(graph[start_vertex], key=lambda x: graph[start_vertex][x]['weight'])
        
        # Try each edge from the start vertex
        for smallest_neighbor in edges_from_start:
            # Initialize for each attempt
            mst_edges = []
            visited = set()
            priority_queue = []

            # Force the starting vertex to have degree 1 with current edge
            visited.add(start_vertex)
            mst_edges.append((start_vertex, smallest_neighbor, graph[start_vertex][smallest_neighbor]['weight']))
            visited.add(smallest_neighbor)

            # Add edges from the chosen neighbor (not the start vertex) to the queue
            for neighbor in graph[smallest_neighbor]:
                if neighbor not in visited:
                    weight = graph[smallest_neighbor][neighbor]['weight']
                    heapq.heappush(priority_queue, (weight, smallest_neighbor, neighbor))

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

def get_top_cocycles_data(ph_output: dict, n: int = 5, method:str = "ripser") -> list:
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
    list
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
    
    cocycle_data = []
    for i in top_n_indices:
        cocycle = ph_output['cocycles'][1][i] if method == "ripser" else ph_output.cocycles_[1][i]
        birth_distance = h1_dgm[i, 0]
        u, v, coeff = cocycle[0]
        critical_edge = (u, v)
        pers = persistence[i]
        cocycle_data.append(
            {"birth_dist":birth_distance, "cocycle":cocycle, "critical_edge":critical_edge, "pers": pers}
        )

    # Sort the cocycles by birth distance in descending order
    # cocycle_data.sort(key=lambda x: x["birth_distance"], reverse=True)
    return cocycle_data

def get_all_loop_nodes(top_cocycles_data, points):
    """
    Returns a list of nodes in all loops from the top cocycles.

    Params
    --------
    top_cocycles_data (list)
        The top cocycles extracted from the ripser result.
    points (np.ndarray)
        The point cloud.

    Returns
    --------
    list
        A list of nodes in all loops.
    """

    birth_distance, _, edge, _ = top_cocycles_data[0]
    sparse_G = vietoris_rips_graph(points, birth_distance)
    _, cycle = prim_tree_find_loop(sparse_G, edge, points)
    reps = [cycle]

    for i in range(1, len(top_cocycles_data)):
        birth_distance, _, edge, _ = top_cocycles_data[i]
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
    return all_data[unique_inds], unique_inds

def critical_edge_method(
    data:np.ndarray, ph:dict=None, n_loops:int = 1, verbose:bool=False, method:str = "ripser"
    )->list: 
    """
    Returns a list homology data for `n_loops` with largest lifetimes in Dgm_1(data). 

    Params
    ------
    data(np.ndarray)
        Input dataset to extract largest cycle from. 
    
    ph (dict, optional)
        Precomputed PH output from ripser. If None, program will compute PH from scratch.
    
    n_loops(int)

    verbose(bool, default=False)
        If set to True, print statements of algorithm steps will be delidered. 
    
    method (str, default = "ripser")
        One of [`ripser`, `dreimac`]. Specifies the type of PH output.        
    Returns
    -------
    top_cocycle_data (list)
        List of `n_loops` dictionaries containing topological information. 
        The item for key "loop" is a size (n,2) numpy array containing the edges of the 
        topological loop with largest lifetime in the PH computation.
    """
    assert method in ["ripser", "dreimac"], f"Method {method} not supported. Choose one of ['ripser', 'dreimac']"

    if ph == None:
        if verbose: print("Starting de novo PH computation...")
        ph = ripser(data, do_cocycles=True)
        if verbose:print("Finished computing PH.")

    top_cocycle_data= get_top_cocycles_data(ph,n=n_loops, method=method)

    iterable = range(n_loops) if n_loops==1 else tqdm(range(n_loops))
    
    for i in iterable:
        birth_dist=top_cocycle_data[i]["birth_dist"]
        crit_edge=top_cocycle_data[i]["critical_edge"]
        if verbose:print("Starting VR graph construction.")
        one_skeleton = vietoris_rips_graph(data, birth_dist)
        if verbose:print(f"Finished VR graph. Starting {i+1}-th loop discovery...")
        _, topological_loop = prim_tree_find_loop(one_skeleton, crit_edge, data)
        if verbose:print("Finished computing loop from VR graph.")
        topological_loop = np.array(topological_loop)
        top_cocycle_data[i]["loop"] = topological_loop
    
    if n_loops==1:
        return [top_cocycle_data[0]]
        #topo_loops= topo_loops[0]

    if verbose: print("Finished critical edge algorithm.")
    return top_cocycle_data[:n_loops]