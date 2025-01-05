from plotly import graph_objects as go
from plotly.colors import qualitative

def plot_all_loops_3d(
    data, cycle_list, birth_times, life_times,
    title=None, pcs_viz=(0, 1, 2), color_col=None, hover_cols=None, pc_prefix="pc"
):
    """
    Returns a plotly interactive figure with all loops in 3D.

    Params
    ------
    data (pd.DataFrame)
        DataFrame with point cloud.

    cycle_list (list of lists)
        List of cycles, where each cycle is a list of edges (tuples).
    
    birth_times (list of float)
        Birth times of the cycles.
    
    life_times (list of float)
        Life times of the cycles.

    title (str, optional)
        Title of the plot. Default is None.
    
    pcs_viz (tuple, optional)
        Indices of principal components to visualize. Default is (0, 1, 2).
    
    color_col (str, optional)
        Column name for categorical coloring. Default is None.
    
    hover_cols (list, optional)
        List of column names to include in the hover information. Default is None.
    
    pc_prefix (str, optional)
        Prefix for principal component labels. Default is "pc".
    
    Returns
    -------
    fig (plotly.graph_objects.Figure)
        A Plotly Figure object containing the 3D scatter plot with loops.
    """
    # Dynamically set x, y, z columns
    x_col = f"{pc_prefix}{pcs_viz[0]}"
    y_col = f"{pc_prefix}{pcs_viz[1]}"
    z_col = f"{pc_prefix}{pcs_viz[2]}"

    pal = ["#000", "#d95f02", "#7570b3", "#a6761d", "#666666"]

    # Create the node trace using annotated_scatter_3d
    node_trace = annotated_scatter_3d(data, x_col, y_col, z_col, color_col, hover_cols)

    # Create edge traces for each cycle
    cycle_traces = []
    for i, cycle in enumerate(cycle_list):
        edge_df = pd.DataFrame(
            [
                {
                    x_col: data.iloc[edge[0]][x_col],
                    y_col: data.iloc[edge[0]][y_col],
                    z_col: data.iloc[edge[0]][z_col],
                    "x_end": data.iloc[edge[1]][x_col],
                    "y_end": data.iloc[edge[1]][y_col],
                    "z_end": data.iloc[edge[1]][z_col],
                }
                for edge in cycle
            ]
        )

        # Add lines connecting the start and end points of each edge
        cycle_traces.append(
            go.Scatter3d(
                x=pd.concat([edge_df[x_col], edge_df["x_end"], pd.Series([None] * len(edge_df))]),
                y=pd.concat([edge_df[y_col], edge_df["y_end"], pd.Series([None] * len(edge_df))]),
                z=pd.concat([edge_df[z_col], edge_df["z_end"], pd.Series([None] * len(edge_df))]),
                mode="lines",
                line=dict(color=pal[i % len(pal)], width=5),
                name=f"Cycle {i + 1} (Birth: {round(birth_times[i], 2)}, Lifetime: {round(life_times[i], 2)})",
            )
        )

    # Create the figure with node trace and cycle traces
    fig = go.Figure(data=[node_trace] + cycle_traces)
    fig.update_layout(
        title=title,
        showlegend=True,
        width=1000,
        height=800,
        scene=dict(
            xaxis_title=f"{pc_prefix} {pcs_viz[0]}",
            yaxis_title=f"{pc_prefix} {pcs_viz[1]}",
            zaxis_title=f"{pc_prefix} {pcs_viz[2]}",
        ),
    )
    
    #fig.show()
    return fig


def annotated_scatter_3d(df, x_col, y_col, z_col, color_col=None, hover_cols=None):
    """
    Params
    ------
    df (pd.DataFrame)
        Input DataFrame containing the data for the plot.
    
    x_col (str)
        Column name for x-coordinates.
    
    y_col (str)
        Column name for y-coordinates.
    
    z_col (str)
        Column name for z-coordinates.
    
    color_col (str, optional)
        Column name for categorical coloring. If None, a default color will be used.
    
    hover_cols (list, optional)
        List of column names to include in the hover information. If None, hover data will not be displayed.
    
    Returns
    -------
    scatter (plotly.graph_objects.Scatter3d)
        A Plotly Scatter3d object for the 3D scatter plot.
    """
    # Initialize hover information if hover_cols is provided
    if hover_cols:
        df['hover_info'] = df.apply(
            lambda row: '<br>'.join([f"{col}: {row[col]}" for col in hover_cols]),
            axis=1
        )
    else:
        df['hover_info'] = None  # No hover information by default

    # Initialize colors based on the color_col if provided
    if color_col:
        categories = df[color_col].unique()
        color_map = {category: qualitative.D3[i % len(qualitative.D3)] for i, category in enumerate(categories)}
        df['color'] = df[color_col].map(color_map)
    else:
        df['color'] = "#D3D3D3" # Default color

    # Create the Scatter3d object
    scatter = go.Scatter3d(
        x=df[x_col],  # x-coordinates
        y=df[y_col],  # y-coordinates
        z=df[z_col],  # z-coordinates
        mode='markers',  # Marker style
        marker=dict(size=1, color=df['color'], opacity=0.8),  # Marker properties
        text=df['hover_info'],  # Combined hover data
        hoverinfo='text' if hover_cols else 'skip'  # Show hover only if hover_cols is provided
    )

    return scatter
