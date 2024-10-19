from matplotlib.colors import LinearSegmentedColormap, to_rgb

def hexlist_to_mpl_cmap(hexlist): 
    """Returns matplotlib colormap given a hexlist.
    """
    cmap_rgb = [to_rgb(x) for x in hexlist]
    cmap = LinearSegmentedColormap.from_list("foo", cmap_rgb)
    return cmap

def inna_palette():
    return ["#65bec3", "#94c77f", "#f06341", "#642870", "#35b779", "#d1cf5e", "#4572ab", "#f58669", ]

def caltech_palette(): 
    return ["#000", "#d95f02", "#7570b3", "#a6761d", "#666666"]

def cat_color_list():
    return ['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','#f0027f','#bf5b17','#666666']
