'''
This Python module is for creating plots of 30-node functional connectivity networks. Node positions are based on their corresponding EEG channel locations.
The code can be used to display multiple network properties at once and highlight nodes of interest. It is used directly for some of the figures seen in [1],
particularly for displaying trophic levels and changes in coherence. The plot format was inspired by Slowinski et al [3].

If you make use of code provided here please site [1] and [3].

Functions included are:
- single_trophic_measures : returns a list of single trophic properties (from one network), such as incoherence, trophic range, and trophic mean 
- double_trophic_measures : returns a list of double trophic properties (for comparing two networks), such as level change and correlaton

Refs:
 [1] Andrews (2025), "Exploring Trophic Analysis for Directed Biological Networks"
 [3] Slowinski et al. (2019), "Background EEG connectivity captures the time-course of epileptogenesis in a mouse model of epilepsy", eNeuro

From: Sarah Andrews
'''


################################
###                          ###
###          IMPORTS         ###
###                          ###
################################

import matplotlib.pyplot as plt
from operator import itemgetter
import matplotlib as mpl


################################
###                          ###
###    INTERNAL FUNCTIONS    ###
###                          ###
################################

# Make a list of weighted edges from network input
def _make_weighted_edges(network):
    network_length = len(network)
    weighted_edges = []
    for i in range(0, network_length):
        for j in range(0, network_length):
            if network[i][j] != 0:
                weighted_edges.append((i+1, j+1, network[i][j]))
    return weighted_edges

# Find node locations based on electode positions
def _make_locations():
    per_row = [6,7,6,5,4,3]
    coords = []
    width = 1
    for i in range(0,6):
        y = i-1
        n = per_row[i]
        if n%2 == 1:
            start = -width*(n-1)/2
        else:
            start = -width*n/2 + width/2
        for j in range (0,n):
            x = start + j*width
            coords.append((x,y))
    coords.pop(9)
    node_nums = [20,18,16,1,2,4,21,19,17,3,5,6,25,24,23,22,8,9,26,27,7,10,11,28,29,12,13,30,14,15]
    corresponding = sorted([(a, b) for a, b in zip(node_nums, coords)])
    ordered_coords = [j for i,j in corresponding]
    return ordered_coords


################################
###                          ###
###     USABLE FUNCTIONS     ###
###                          ###
################################

def mouse_subplots(networks,
                   values,
                   plot_layout,
                   main_title='',
                   sub_titles=None,
                   with_arrows=False,
                   take_absolute_value=0,
                   show_imbalance=False,
                   highlight_nodes=None,
                   node_size_range=[5, 26],
                   scale_values_with=[0, 0.5, 1]):
    ''' 
    This function takes list networks and node values as inputs, and plots a figure containing subplots.
    INPUTS:
      networks : list of adjacency matrices (which are themselves given as a list of lists)
      values : list of node value lists (each node value list has values ordered for node 1 to 30)
      plot_layout : list containing number of rows and columns
      main_title : figure title
      sub_titles : list of sub titles for plots
      with_arrows : (bool) optionally include arrow heads
      take_absolute_value : 0, 1, or 2.
          0 = do not use absolute value
          1 = use absolute value but dont make colours depend on it
          2 = use and plot colours for absolute value
      show_imbalance : (bool) display node imbalance in colouring
      highlight_nodes : optionally highlight a given list of nodes in plot
      node_size_range : control minimum and maximum node size
      scale_values_with : scale values for node size and colour according to a [minimum, middle, maximum]
    OUTPUTS:
      The required plot is displayed
    '''
    
    mpl.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.bottom'] = False

    # Set figure size depending on existance of titles
    if sub_titles == None:
        sub_titles = [[None]*plot_layout[1]]*plot_layout[0]
        fig, axs = plt.subplots(plot_layout[0],plot_layout[1],figsize=(plot_layout[1]*3-1,plot_layout[0]*2+1))
    else:
        fig, axs = plt.subplots(plot_layout[0],plot_layout[1],figsize=(plot_layout[1]*3,plot_layout[0]*2+1.5))

    if highlight_nodes == None:
        highlight_nodes = [[None]*plot_layout[1]]*plot_layout[0]

    # Generate subplots              
    for i in range(0, plot_layout[0]):
        for j in range(0, plot_layout[1]):
            mouse_plot(networks[i][j], values[i][j], title=sub_titles[i][j], show_imbalance=show_imbalance, node_size_range=node_size_range, scale_values_with=scale_values_with, with_arrows=with_arrows, take_absolute_value=take_absolute_value, subplot=axs[i,j], highlight_nodes=highlight_nodes[i][j])

    plt.suptitle(main_title,fontsize=13,fontweight="bold")
    plt.show()
    

def mouse_plot(network,
               values,
               title=None,
               with_arrows=False,
               take_absolute_value=0,
               show_imbalance=False,
               highlight_nodes=None,
               node_size_range=[5, 26],
               scale_values_with=[0, 0.5, 1],
               subplot=None):
    ''' 
    This function take a network and node value list as inputs, and produces a plot.
    INPUTS:
      network : adjacency matrix (list of lists)
      values : list of node values (values ordered for node 1 to 30)
      title : title for plot
      with_arrows : (bool) optionally include arrow heads
      take_absolute_value : 0, 1, or 2.
          0 = do not use absolute value
          1 = use absolute value but dont make colours depend on it
          2 = use and plot colours for absolute value
      show_imbalance : (bool) display node imbalance in colouring
      highlight_nodes : optionally highlight a given list of nodes in plot
      node_size_range : control minimum and maximum node size
      scale_values_with : scale values for node size and colour according to a [minimum, middle, maximum]
      subplot : optionally give an axis to use
    OUTPUTS:
      The required plot is displayed
    '''
    
    net_len = len(network)
    [min_with, mean_with, max_with] = scale_values_with
    
    # Find the heaviest 6.5% of edge weights
    weights = _make_weighted_edges(network)
    heaviest_weights = sorted(weights, key=itemgetter(2))[int(0.935*len(weights)):]

    # Calculate node colours and sizes based on values
    colours = [] ; sizes = []
    for i in range(0, net_len):
        value = values[i]

        # Check if absolute value needed
        if take_absolute_value > 0:
            value = abs(value)
            
        # Find colour multiplier
        if min_with <= value <= mean_with:
            m = 0.5/(mean_with-min_with)*(value-min_with)
        elif mean_with < value <= max_with:
            m = 0.5/(max_with-mean_with)*(value-max_with)+1
        elif value < min_with:
            m = 0
        else:
            m = 1

        # Red and blue colours for node imbalance option
        if show_imbalance:
            out_degree = sum(network[i])
            in_degree = sum([network[x][i] for x in range(0, net_len)])
            if in_degree>=out_degree:
                colour = (0,0,1-m)
            else:
                colour = (1-m,0,0)
                
        # Orange and green colours for absolute value option
        elif take_absolute_value == 2:
            if values[i]>0:
                colour = (0,1-m*0.7,0)
            else:
                colour = (1-m*0.7,(1-m*0.7)/2,0)

        # Standard yellow to red gradient otherwise
        else:
            colour = (1,1-m*0.7,0)
        
        # Calculate size of nodes
        multiplier = (value-min_with)/(max_with-min_with)
        if value > max_with:
            multiplier = 1 + (value-min_with)*0.01 # Allow to continue growing at a tiny rate
        size = node_size_range[0]+multiplier*(node_size_range[1]-8)
        
        colours.append(colour)
        sizes.append(size)

    # Coordinates for nodes (found with _make_node_locations)
    coords = [(0.5, -1), (1.5, -1), (1.0, 0), (2.5, -1), (2.0, 0), (3.0, 0), (0.0, 2), (1.5, 1), (2.5, 1), (1.0, 2), (2.0, 2), (0.5, 3), (1.5, 3), (0.0, 4), (1.0, 4), (-0.5, -1), (-1.0, 0), (-1.5, -1), (-2.0, 0), (-2.5, -1), (-3.0, 0), (0.5, 1), (-0.5, 1), (-1.5, 1), (-2.5, 1), (-2.0, 2), (-1.0, 2), (-1.5, 3), (-0.5, 3), (-1.0, 4)]

    # If not given subplot axis
    if subplot == None:
        mpl.rcParams['font.family'] = 'Arial'
        plt.figure(figsize = (6,5))
        plt.rcParams['axes.spines.left'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.bottom'] = False

        # Plot significant edges
        for i in range(0, len(heaviest_weights)):
            x = heaviest_weights[i][0] ; y = heaviest_weights[i][1]

            # Plot lines
            plt.plot([coords[x-1][0],coords[y-1][0]],[coords[x-1][1],coords[y-1][1]],'-',color=(0.5,0.5,0.5))

            # Add arrow heads if requested
            if with_arrows:
                if (coords[y-1][0] > coords[x-1][0]):
                    sign = 1
                else:
                    sign = -1
                if (coords[y-1][0]-coords[x-1][0]) == 0:
                    plt.arrow(coords[x-1][0]+(coords[y-1][0]-coords[x-1][0])*0.9, coords[x-1][1]+(coords[y-1][1]-coords[x-1][1])*0.9, 0.01*sign, 0.01*sign*(coords[y-1][1]-coords[x-1][1])/(coords[y-1][0]-coords[x-1][0]+0.000001), shape='full', lw=0, length_includes_head=True, head_width=.15, color="grey")
                else:
                    plt.arrow(coords[x-1][0]+(coords[y-1][0]-coords[x-1][0])*0.9, coords[x-1][1]+(coords[y-1][1]-coords[x-1][1])*0.9, 0.01*sign, 0.01*sign*(coords[y-1][1]-coords[x-1][1])/(coords[y-1][0]-coords[x-1][0]), shape='full', lw=0, length_includes_head=True, head_width=.15, color="grey")

        # Plot nodes
        for i in range(0, net_len):
            plt.plot(coords[i][0],coords[i][1],'o',color=colours[i],markersize=sizes[i])
            if isinstance(highlight_nodes, list):
                if i+1 in highlight_nodes:
                    plt.plot(coords[i][0],coords[i][1],'o',color=(0,0,0),markersize=sizes[i]*0.6)

        # Other plot features
        plt.text(0,-2,"min: "+str(round(min(values),4))+"; max: "+str(round(max(values),4)),horizontalalignment='center')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.xlim([-4,4])
        plt.ylim([-2.5,4.7])
        font = {"fontweight": "bold", "fontsize": 13}
        plt.title(title, font)
        plt.show()

    else:
        # Plot significant edges
        for i in range(0, len(heaviest_weights)):
            x = heaviest_weights[i][0] ; y = heaviest_weights[i][1]

            # Plot lines
            subplot.plot([coords[x-1][0],coords[y-1][0]],[coords[x-1][1],coords[y-1][1]],'-',color=(0.5,0.5,0.5))

            # Add arrow heads if requested
            if with_arrows:
                if (coords[y-1][0] > coords[x-1][0]):
                    sign = 1
                else:
                    sign = -1
                if (coords[y-1][0]-coords[x-1][0]) == 0:
                    subplot.arrow(coords[x-1][0]+(coords[y-1][0]-coords[x-1][0])*0.9, coords[x-1][1]+(coords[y-1][1]-coords[x-1][1])*0.9, 0.01*sign, 0.01*sign*(coords[y-1][1]-coords[x-1][1])/(coords[y-1][0]-coords[x-1][0]+0.000001), shape='full', lw=0, length_includes_head=True, head_width=.15, color="grey")
                else:
                    subplot.arrow(coords[x-1][0]+(coords[y-1][0]-coords[x-1][0])*0.9, coords[x-1][1]+(coords[y-1][1]-coords[x-1][1])*0.9, 0.01*sign, 0.01*sign*(coords[y-1][1]-coords[x-1][1])/(coords[y-1][0]-coords[x-1][0]), shape='full', lw=0, length_includes_head=True, head_width=.15, color="grey")

        # Plot nodes
        for i in range(0, net_len):
            subplot.plot(coords[i][0],coords[i][1],'o',color=colours[i],markersize=sizes[i])
            if isinstance(highlight_nodes, list):
                if i+1 in highlight_nodes:
                    subplot.plot(coords[i][0],coords[i][1],'o',color=(0,0,0),markersize=sizes[i]*0.6)

        # Other plot features
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.text(0,-2,"min: "+str(round(min(values),4))+"; max: "+str(round(max(values),4)),horizontalalignment='center')
        subplot.set_xlim([-4,4])
        subplot.set_ylim([-2.5,4.7])
        if title != None:
            subplot.set_title(title)
