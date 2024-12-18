from plotly import graph_objects as go

def plot_all_loops_3d(point_cloud, cycle_list, birth_times, life_times, title=None, pcs_viz=(0, 1, 2)):
    """
    Plot all loops in 3D with the critical edges.

    Parameters:
    - point_cloud (np.ndarray): The point cloud.
    - critical_edge_list (list): The list of critical edges.
    - cycle_list (list): The list of cycles.
    - birth_times (list): The birth times of the cycles.
    - life_times (list): The life times of the cycles.
    - title (str, optional): The title of the plot, by default None.
    - pcs_viz (tuple, optional): The principal components to visualize, by default (0, 1, 2).
    """
    pal = ["#000", "#d95f02", "#7570b3", "#a6761d", "#666666"]

    positions = {i: point_cloud[i, pcs_viz] for i in range(len(point_cloud))}
    node_trace = go.Scatter3d(
        x=[positions[i][0] for i in positions],
        y=[positions[i][1] for i in positions],
        z=[positions[i][2] for i in positions],
        mode='markers', marker=dict(size=1, color="#D3D3D3", opacity=0.7),
        name='Nodes')

    cycle_traces = []
    for i, cycle in enumerate(cycle_list):
        x, y, z = [], [], []
        for edge in cycle:
            x.extend([positions[edge[0]][0], positions[edge[1]][0], None])
            y.extend([positions[edge[0]][1], positions[edge[1]][1], None])
            z.extend([positions[edge[0]][2], positions[edge[1]][2], None])
        cycle_traces.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', 
                            line=dict(color=pal[i % len(pal)], width=5), 
                            name=f'Cycle {i+1} (Birth: {round(birth_times[i], 2)}, Lifetime: {round(life_times[i], 2)})')
        )

    fig = go.Figure(data = [node_trace] + cycle_traces)
    fig.update_layout(title=title, showlegend=True, width=1000, height=800, 
                      scene=dict(xaxis_title=f"pc {pcs_viz[0]}", yaxis_title= f"pc {pcs_viz[1]}", zaxis_title=f"pc {pcs_viz[2]}"))
    fig.show()