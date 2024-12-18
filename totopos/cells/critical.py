import numpy as np
import anndata as ad
from os.path import join
import math
import pandas as pd
import time
from ripser import ripser
from persim import plot_diagrams
import networkx as nx
import heapq
from sklearn.neighbors import BallTree

def prim_tree_find_loop(graph, critical_edge, points):
    """
    Construct a tree where the starting vertex has degree 1.
    After adding the critical critical_edge, check if a cycle is formed using NetworkX.
    Loops over multiple possible starting vertices.

    Parameters:
    - graph (dict): The graph as an adjacency list.
    - critical_edge (tuple): The critical critical_edge to add to the tree.
    - points (np.ndarray): The points in the graph.

    Returns:
    - (tuple): A tuple of the MST critical_edges and the cycle formed by adding the critical critical_edge.
    """
    u_edge, v_edge = critical_edge[0], critical_edge[1]

    for start_vertex in [u_edge, v_edge]:
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

def vietoris_rips_graph(point_cloud, birth_time, epsilon=1e-4):
    '''
    Create a Vietoris-Rips graph from a point cloud using the birth time as = a threshold.
    
    Parameters:
    - point_cloud (np.ndarray): The point cloud to create the graph from.
    - birth_time (float): The birth distance threshold of the cocycle.
    - epsilon (float, optional): For numerical stability, by default 1e-4.

    Returns:
    - nx.Graph: The sparse graph.
    '''
    
    threshold = birth_time + epsilon

    G = nx.Graph()
    num_points = len(point_cloud)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = np.linalg.norm(point_cloud[i] - point_cloud[j])
            if dist <= threshold:
                G.add_edge(i, j, weight=dist)
    return G


def get_top_cocycles_data(ph_output_ripser, n=5):
    """
    Returns the top n most persistent cocycles, their birth times, critical edges, and life times from ripser PH computation, 
    sorted by persistence in descending order.

    Parameters:
    - ph_output_ripser (dict): The ph_output_ripser dictionary from ripser containing 'dgms' and 'cocycles'.
    - n (int): The number of top persistent cocycles to extract.

    Returns:
    - list: A list of dictionaries containing the birth time, cocycle, critical edge, and persistence.
        The cocycles are sorted by persistence in descending order.
    """

    # Extract the persistence diagram for H1 (1-dimensional features)
    h1_dgm = ph_output_ripser['dgms'][1]
    
    # Calculate persistence (death - birth) and sort by persistence in descending order
    persistence = h1_dgm[:, 1] - h1_dgm[:, 0]
    sorted_indices = np.argsort(persistence)[::-1]
    
    # Get the top n most persistent cocycles
    top_n_indices = sorted_indices[:n]
    
    cocycle_data = []
    for i in top_n_indices:
        cocycle = ph_output_ripser['cocycles'][1][i]
        birth_time = h1_dgm[i, 0]
        u, v, coeff = cocycle[0]
        critical_edge = (u, v)
        pers = persistence[i]
        cocycle_data.append(
            {"birth_time":birth_time, "cocycle":cocycle, "critical_edge":critical_edge, "pers": pers}
        )

    # Sort the cocycles by birth time in descending order
    # cocycle_data.sort(key=lambda x: x["birth_time"], reverse=True)
    return cocycle_data

def get_all_loop_nodes(top_cocycles_data, points):
    """
    Extract the set of nodes in all loops from the top cocycles
    by finding the loop with the critical edge in the spanning tree.
    The sparse graph is constructed only once using the cocycle with 
    the highest birth distance threshold for efficiency.

    Parameters:
    - top_cocycles (list): The top cocycles extracted from the ripser result.
    - points (np.ndarray): The point cloud.

    Returns:
    - list: A list of nodes in all loops.
    """

    birth_time, _, edge, _ = top_cocycles_data[0]
    sparse_G = vietoris_rips_graph(points, birth_time)
    _, cycle = prim_tree_find_loop(sparse_G, edge, points)
    reps = [cycle]

    for i in range(1, len(top_cocycles_data)):
        birth_time, _, edge, _ = top_cocycles_data[i]
        _, cycle = prim_tree_find_loop(sparse_G, edge, points)
        reps.append(cycle)

    cycle_nodes = set()
    for cycle in reps:
        for edge in cycle:
            cycle_nodes.update(edge)

    return list(cycle_nodes)

def get_loop_inds(all_labels, cycle_nodes, unique_labels):
    """Locate the indices of the loop nodes in the original data."""
    return np.where(np.isin(all_labels, unique_labels[cycle_nodes]))[0]

def get_loop_coords(cycle, all_labels, u_labels, orig_data):
    """
    Find the coordinates of the loop nodes in the original data.

    Parameters:
    - cycle (list): The loop to extract nodes from.
    - all_labels (np.ndarray): The cluster labels of the original data.
    - u_labels (np.ndarray): The unique labels.
    - orig_data (np.ndarray): The original data.

    Returns:
    - np.ndarray: The coordinates of the loop nodes in the original data.
    """
    cycle_nodes = set()
    for edge in cycle:
        cycle_nodes.update(edge)
    cycle_nodes = list(cycle_nodes)
    cycle_inds = np.where(np.isin(all_labels, u_labels[cycle_nodes]))[0]
    return orig_data[cycle_inds]

def get_loop_neighbors(all_data, query_data, radius, leaf_size=40):
    """
    Get the neighbors of the loop nodes in the original data using a BallTree.
    
    Parameters:
    - all_data (np.ndarray): The original data.
    - query_data (np.ndarray): The loop nodes.
    - radius (float): The radius of the neighborhood.
    - leaf_size (int, optional): The leaf size of the BallTree, by default 40
    (smaller leaf size is faster but uses more memory).
    
    Returns:
    - tuple: A tuple of the neighbor data and indices.
    """
    tree = BallTree(all_data, leaf_size=leaf_size)
    inds = tree.query_radius(query_data, r=radius)
    unique_inds = np.unique(np.concatenate(inds))
    return all_data[unique_inds], unique_inds

def critical_edge_method(data:np.ndarray, ph:dict=None, n_loops:int = 1): 
    """
    Returns homology representative of PH class with largest lifetime in Dgm_1(data). 

    Params
    ------
    data(np.ndarray)
        Input dataset to extract largest cycle from. 
    
    ph (dict, optional)
        Precomputed PH output from ripser. If None, program will compute PH from scratch.

    Returns
    -------
    top_cocycle_data (list)
        List of `n_loops` dictionaries containing topological information. 
        The item for key "loop" is a size (n,2) numpy array containing the edges of the 
        topological loop with largest lifetime in the PH computation.
    """
    if ph == None:
        ph = ripser(data, do_cocycles=True)
    top_cocycle_data= get_top_cocycles_data(ph,n=n_loops)
    #topo_loops = []
    for i in range(n_loops):
        birth_time=top_cocycle_data[i]["birth_time"]
        crit_edge=top_cocycle_data[i]["critical_edge"]
        one_skeleton = vietoris_rips_graph(data, birth_time)
        _, topological_loop = prim_tree_find_loop(one_skeleton, crit_edge, data)
        topological_loop = np.array(topological_loop)
        #topo_loops.append(topological_loop)
        top_cocycle_data[i]["loop"] = topological_loop
    
    if n_loops==1:
        return [top_cocycle_data[0]]
        #topo_loops= topo_loops[0]

    return top_cocycle_data[:n_loops]