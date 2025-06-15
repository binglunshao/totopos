from plotly import graph_objects as go
from plotly.colors import qualitative
from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd
import copy
import seaborn as sns
from .palettes import cat_color_list
cat = np.concatenate

def inna_palette():
    return ["#65bec3", "#94c77f", "#f06341", "#642870", "#35b779", "#d1cf5e", "#4572ab", "#f58669", ]

def caltech_palette(): 
    return ["#000", "#d95f02", "#7570b3", "#a6761d", "#666666"]

def cat_color_list():
    return ['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','#f0027f','#bf5b17','#666666']

def plot_all_loops_3d(
    data, cycle_list, birth_times, life_times,
    title=None, pcs_viz=(0, 1, 2), color_col=None, 
    hover_cols=None, pc_prefix="pc", white_background:bool=True,
    palette_scatter=None,
    dot_size=10
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
    
    white_background (bool, optional)
        If True, the background will be white. Default is True.
    
    palette_scatter (str, optional)
        Color palette for the scatter plot. Default is None.
    
    Returns
    -------
    fig (plotly.graph_objects.Figure)
        A Plotly Figure object containing the 3D scatter plot with loops.
    """
    # Dynamically set x, y, z columns
    x_col = f"{pc_prefix}{pcs_viz[0]}"
    y_col = f"{pc_prefix}{pcs_viz[1]}"
    z_col = f"{pc_prefix}{pcs_viz[2]}"

    pal = ["#000", "#d95f02", "#7570b3", "#a6761d", '#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','#f0027f','#bf5b17','#666666']

    # Create the node trace using annotated_scatter_3d
    node_trace = annotated_scatter_3d(
        data, x_col, y_col, z_col, color_col=color_col, hover_cols=hover_cols, cmap=palette_scatter,
        dot_size=dot_size
    )

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

        layout_options = {
            "title": title,
            "showlegend": True,
            "width": 1000,
            "height": 800,
            "scene": dict(
                xaxis_title=f"{pc_prefix} {pcs_viz[0]}",
                yaxis_title=f"{pc_prefix} {pcs_viz[1]}",
                zaxis_title=f"{pc_prefix} {pcs_viz[2]}",
            ),
        }

    # Apply white background if specified
    if white_background:
        layout_options.update({
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
            "scene": dict(
                xaxis=dict(backgroundcolor="white"),
                yaxis=dict(backgroundcolor="white"),
                zaxis=dict(backgroundcolor="white"),
            ),
        })
    fig.update_layout(**layout_options)
    return fig 


def annotated_scatter_3d(df_, x_col, y_col, z_col, color_col=None, hover_cols=None, cmap=None, alpha=0.7, dot_size=1):
    """
    3D scatter plot with optional categorical coloring.

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

    cmap (list, optional)
        Custom list of hex color codes for the scatter points (categorical palette).

    alpha (float, default=0.7)
        Opacity of scatter dots.

    dot_size (int, default=1)
        Size of scatter dots.

    Returns
    -------
    scatter (plotly.graph_objects.Scatter3d)
        A Plotly Scatter3d object for the 3D scatter plot.
    """
    df = df_.copy()
    # Default color palette for categorical data
    default_palette = ["#65bec3", "#94c77f", "#f06341", "#642870", "#35b779", "#d1cf5e", "#4572ab", "#f58669"]
    
    # Use provided cmap if supplied, otherwise default to default_palette
    color_palette = cmap if cmap else default_palette

    # Initialize hover information if hover_cols is provided
    if hover_cols is not None:
        df['hover_info'] = df.apply(
            lambda row: '<br>'.join([f"{col}: {row[col]}" for col in hover_cols]),
            axis=1
        )
    else:
        df['hover_info'] = None  # No hover information by default

    # Initialize colors for categorical data
    if color_col is not None:
        if isinstance(df[color_col].dtype, pd.CategoricalDtype):
            df[color_col] = df[color_col].astype(str)

        # Map categories to colors
        categories = df[color_col].unique()
        color_map = {category: color_palette[i % len(color_palette)] for i, category in enumerate(categories)}
        df['color'] = df[color_col].map(color_map)
    else:
        df['color'] = "#D3D3D3"  # Default color if no color_col provided

    # Create the Scatter3d object
    scatter = go.Scatter3d(
        x=df[x_col],  # x-coordinates
        y=df[y_col],  # y-coordinates
        z=df[z_col],  # z-coordinates
        mode='markers',  # Marker style
        marker=dict(size=dot_size, color=df['color'], opacity=alpha),  # Marker properties
        text=df['hover_info'],  # Combined hover data
        hoverinfo='text' if hover_cols else 'skip',  # Show hover only if hover_cols is provided
        showlegend=False  # Suppress legend for scatter points
    )

    return scatter

def plot_loops(
    adata, 
    topological_loops,
    title=None, 
    pcs_viz=(0, 1, 2), 
    color_col=None, 
    hover_cols=None, 
    white_background=True,
    palette_scatter=None,
    dot_size=1,
    line_width=4,
    use_pca=True
):
    """
    Plot topological loops overlaid on single-cell data from AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing single-cell data
    topological_loops : list of dict
        List of dictionaries containing loop information with keys:
        - 'loop': list of tuples representing edges
        - 'birth_dist': birth time of the cycle  
        - 'pers': persistence/lifetime of the cycle
        - 'topocell_ixs': array of indices mapping to original adata
    title : str, optional
        Title of the plot
    pcs_viz : tuple, optional
        Indices of principal components to visualize (default: (0, 1, 2))
    color_col : str, optional
        Column name in adata.obs for categorical coloring
    hover_cols : list, optional
        List of column names from adata.obs to include in hover info
    white_background : bool, optional
        Whether to use white background (default: True)
    palette_scatter : str, optional
        Color palette for scatter plot
    dot_size : int, optional
        Size of scatter plot points (default: 3)
    line_width : int, optional
        Width of loop lines (default: 4)
    use_pca : bool, optional
        Whether to use PCA coordinates from adata.obsm['pcs'] (default: True)
        If False, will create DataFrame from topocells

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D plot with loops overlaid on cell data
    """
    
    # Color palette for loops
    loop_palette = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", 
        "#ffff33", "#a65628", "#f781bf", "#999999", "#1f78b4",
        "#33a02c", "#fb9a99", "#cab2d6", "#6a3d9a", "#b15928"
    ]
    
    if use_pca and 'pcs' in adata.obsm:
        # Use PCA coordinates from AnnData
        pca_coords = adata.obsm['pcs']
        
        # Create DataFrame with PCA coordinates
        n_pcs = min(pca_coords.shape[1], max(pcs_viz) + 1)
        pc_columns = [f"pc{i}" for i in range(n_pcs)]
        data_df = pd.DataFrame(pca_coords[:, :n_pcs], columns=pc_columns)
        
        # Add obs data for coloring and hover
        for col in adata.obs.columns:
            data_df[col] = adata.obs[col].values
            
    else:
        # Use topocells from the first loop (assuming they represent the full dataset)
        topocells = topological_loops[0]['topocells']
        n_components = topocells.shape[1]
        column_names = [f"pc{i}" for i in range(n_components)]
        data_df = pd.DataFrame(topocells, columns=column_names)
        
        # Map back to original adata indices if available
        if 'topocell_ixs' in topological_loops[0]:
            topocell_ixs = topological_loops[0]['topocell_ixs']
            # Add obs data for the subset of cells
            for col in adata.obs.columns:
                mapped_values = np.full(len(data_df), np.nan, dtype=object)
                if len(topocell_ixs) == len(data_df):
                    mapped_values = adata.obs.iloc[topocell_ixs][col].values
                data_df[col] = mapped_values
    
    # Set coordinate columns
    x_col = f"pc{pcs_viz[0]}"
    y_col = f"pc{pcs_viz[1]}"  
    z_col = f"pc{pcs_viz[2]}"
    
    # Create hover text
    hover_text = None
    if hover_cols:
        available_hover_cols = [col for col in hover_cols if col in data_df.columns]
        if available_hover_cols:
            hover_text = data_df[available_hover_cols].apply(
                lambda row: '<br>'.join([f"{col}: {row[col]}" for col in available_hover_cols]),
                axis=1
            )
    
    # Handle colors
    color_values = None
    if color_col in data_df.columns:
        color_values = data_df[color_col]
        # Convert categorical to string if needed
        if hasattr(color_values, 'cat'):
            color_values = color_values.astype(str)
    else:
        # Default color if no color_col provided
        color_values = np.full(len(data_df), "#D3D3D3")
    
    # Create base scatter plot
    scatter_trace = go.Scatter3d(
        x=data_df[x_col],
        y=data_df[y_col], 
        z=data_df[z_col],
        mode='markers',
        marker=dict(
            size=dot_size,
            color=color_values,
            colorscale=palette_scatter,
            opacity=0.3,
            line=dict(width=0)
        ),
        text=hover_text,
        hoverinfo='text' if hover_text is not None else 'x+y+z',
        name='Cells',
        showlegend=False
    )
    
    # Create loop traces
    loop_traces = []
    
    for i, loop_info in enumerate(topological_loops):
        loop_edges = loop_info['loop']
        birth_time = float(loop_info['birth_dist'])
        persistence = float(loop_info['pers'])
        
        # Get the topocell indices for this loop
        if 'topocell_ixs' in loop_info:
            topocell_ixs = loop_info['topocell_ixs']
            
            # Create mapping from topocell index to data_df index
            if use_pca:
                # Direct mapping - topocell_ixs are original adata indices
                idx_mapping = {orig_idx: orig_idx for orig_idx in topocell_ixs if orig_idx < len(data_df)}
            else:
                # topocell_ixs map to positions in the topocells array
                idx_mapping = {topocell_ixs[j]: j for j in range(len(topocell_ixs))}
        else:
            # Fallback: assume direct indexing
            idx_mapping = {j: j for j in range(len(data_df))}
        
        # Build coordinates for the loop
        x_coords = []
        y_coords = []
        z_coords = []
        
        valid_edges = []
        for edge in loop_edges:
            start_idx = int(edge[0])
            end_idx = int(edge[1])
            
            # Map to data_df indices
            start_mapped = idx_mapping.get(start_idx)
            end_mapped = idx_mapping.get(end_idx)
            
            if (start_mapped is not None and end_mapped is not None and 
                start_mapped < len(data_df) and end_mapped < len(data_df)):
                
                # Add edge coordinates
                x_coords.extend([data_df.iloc[start_mapped][x_col], data_df.iloc[end_mapped][x_col], None])
                y_coords.extend([data_df.iloc[start_mapped][y_col], data_df.iloc[end_mapped][y_col], None])  
                z_coords.extend([data_df.iloc[start_mapped][z_col], data_df.iloc[end_mapped][z_col], None])
                valid_edges.append((start_idx, end_idx))
        
        if len(valid_edges) > 0:
            loop_trace = go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='lines',
                line=dict(
                    color=loop_palette[i % len(loop_palette)], 
                    width=line_width
                ),
                name=f'Loop {i+1} (B:{birth_time:.2f}, P:{persistence:.2f})',
                hoverinfo='name'
            )
            loop_traces.append(loop_trace)
        else:
            print(f"Warning: Loop {i+1} has no valid edges that map to the data")
    
    # Create figure
    fig = go.Figure(data=[scatter_trace] + loop_traces)
    
    # Update layout
    layout_kwargs = {
        'title': title or 'Topological Loops on Single-Cell Data',
        'showlegend': True,
        'width': 1000,
        'height': 800,
        'scene': dict(
            xaxis_title=f'PC{pcs_viz[0]}',
            yaxis_title=f'PC{pcs_viz[1]}', 
            zaxis_title=f'PC{pcs_viz[2]}',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        )
    }
    
    if white_background:
        layout_kwargs.update({
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white'
        })
        layout_kwargs['scene'].update({
            'xaxis': dict(backgroundcolor='white', gridcolor='lightgray'),
            'yaxis': dict(backgroundcolor='white', gridcolor='lightgray'),
            'zaxis': dict(backgroundcolor='white', gridcolor='lightgray')
        })
    
    fig.update_layout(**layout_kwargs)
    
    return fig


def quick_loop_summary(topological_loops):
    """
    Print a quick summary of the topological loops.
    """
    print(f"Number of loops: {len(topological_loops)}")
    print("\nLoop summary:")
    for i, loop_info in enumerate(topological_loops):
        n_edges = len(loop_info['loop'])
        birth = loop_info['birth_dist']
        pers = loop_info['pers']
        n_cells = loop_info['topocells'].shape[0] if 'topocells' in loop_info else 'N/A'
        print(f"  Loop {i+1}: {n_edges} edges, {n_cells} cells, birth={birth:.3f}, persistence={pers:.3f}")


# Example usage:
"""
# Quick summary
quick_loop_summary(topological_loops)

# Plot with AnnData PCA coordinates
fig = plot_topological_loops_anndata(
    adata=adata,
    topological_loops=topological_loops,
    title="Topological Loops in Single-Cell Data",
    pcs_viz=(0, 1, 2),  # Use PC1, PC2, PC3
    color_col='cell_type',  # Color by cell type
    hover_cols=['cell_type', 'sample', 'day'],  # Show in hover
    use_pca=True  # Use PCA coordinates from adata.obsm['pcs']
)

fig.show()

# Alternative: Plot using topocells coordinates
fig2 = plot_topological_loops_anndata(
    adata=adata,
    topological_loops=topological_loops,
    title="Loops in Topological Space", 
    pcs_viz=(0, 1, 2),
    color_col='cell_state',
    use_pca=False  # Use topocells instead of PCA
)

fig2.show()
"""

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


def visualize_h1(data, h1_simplex_list, pal = None, ax = None, d = 2, return_fig = False, alpha = 0.2, scatter = True): 
    assert d in [2, 3], "only 2D and 3D visualizations are supported"
    pal = cat_color_list() if pal is None else pal
    
    if ax is None:
        fig = plt.figure(figsize=(4,4))
        if d == 3:
            ax=fig.add_subplot(projection="3d")
        else: 
            ax=fig.add_subplot()

    n_loops = len(h1_simplex_list)

    for k in range(n_loops):
        for edge in h1_simplex_list[k]: 
            source, tgt= edge
            data_plot=cat([data[np.array([source]), :d], data[np.array([tgt]), :d]], 0)
            ax.plot(*data_plot.T, color = pal[k], linewidth=3)

    if scatter:
        ax.scatter(*data[:, :d].T, s = 1, color = "grey",alpha=alpha)

    ax.azim=50
    ax.axis("off")
    #plt.axis("off")
    if return_fig: return fig

def plot_pers_diag_ripser(dgms:list, ax = None, dot_size = 40, conf_int=None, pal = None):
    """
    Plot persistence diagrams using custom color palette.

    Params
    ------
    dgms (list of np.ndarrays)
        Coordinates for (births, deaths) of each persistent feature across dimensions.
        The i-th list is the persistent diagram of dimension i.
    
    ax (matplotlib.axes._axes.Axes)

    dot_size (int)

    conf_int (float)
    
    pal (list)
    """
    n_dgms = len(dgms)
    
    dgms_, max_val = replace_inf(dgms)
    ax = ax or plt.gca()
    pal = caltech_palette() if pal is None else pal
    
    ax.scatter(
        *dgms_[0].T,
        linewidth=0.1,
        alpha=0.7,
        s = dot_size,
        color=pal[0],
        edgecolors="lightgrey",
        label="$H_0$"
    )

    for i in range(1, n_dgms):
        #plt
        ax.scatter(
            *dgms[i].T,
            linewidth=0.1,
            alpha=0.7,
            s = dot_size,
            color=pal[i],
            edgecolors="lightgrey",
            label= f"$H_{i}$"
        )

    ax.axhline(max_val, linestyle="--", color="grey", label="$\infty$")

    ax.plot([0,max_val+.5], [0, max_val+.5], color = "grey")

    if conf_int is not None: 
        ax.fill_between(x= [0, max_val+.5] , y1= [0, max_val+.5], y2=[conf_int,  max_val + conf_int + .5], alpha = 0.3, color = "lightgrey")

    ax.set_xlim(-.3, max_val)
    ax.set_ylim(0, max_val +.5)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    #plt.legend()
    ax.legend()

def set_plotting_style_plt():

    tw = 1.3
    rc = {'lines.linewidth': 2,
    'axes.labelsize': 18,
    'axes.titlesize': 21,
    'xtick.major' : 12,
    'ytick.major' : 12,
    'xtick.major.width': tw,
    'xtick.minor.width': tw,
    'ytick.major.width': tw,
    'ytick.minor.width': tw,
    'xtick.labelsize': 'large',
    'ytick.labelsize': 'large',
    'font.family': 'sans-serif',
    'weight':'bold',
    'grid.linestyle': ':',
    'grid.linewidth': 1.5,
    'grid.color': '#ffffff',
    'mathtext.fontset': 'stixsans',
    'mathtext.sf': 'fantasy',
    'legend.frameon': True,
    'legend.fontsize': 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.figsize": (3,3),
    "axes.prop_cycle": plt.cycler(color=cat_color_list())
    }



    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('mathtext', fontset='stixsans', sf='sans')
    sns.set_style('ticks', rc=rc)
    sns.set_context('notebook', rc=rc)

def make_inset_axis_label(ax, xlabel="PC 1", ylabel="PC 2"):
    """
    Creates an inset axis for low-dim projection plots.
    """

    inset_ax = ax.inset_axes([0.00, 0.00, 0.25, 0.25])

    inset_ax.set_xlabel(xlabel, fontsize=16)
    inset_ax.set_ylabel(ylabel, fontsize=16)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    # Hide top and right spines for cleaner look
    inset_ax.spines['top'].set_visible(False)
    inset_ax.spines['right'].set_visible(False)
    
    # Optionally, emphasize bottom/left spines
    inset_ax.spines['left'].set_linewidth(0.8)
    inset_ax.spines['bottom'].set_linewidth(0.8)
    inset_ax.set_facecolor('none')