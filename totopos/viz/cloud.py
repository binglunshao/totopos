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