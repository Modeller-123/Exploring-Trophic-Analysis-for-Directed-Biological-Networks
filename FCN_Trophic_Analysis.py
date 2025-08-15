'''
This Python module is for conducting trophic analysis as introduced by MacKay, Johnson and Sansom in [2], on functional connectivity networks.
The code below makes use of the adapted "trophic_tools" [1, 2], and was written by Sarah Andrews for [1]. Many plots were displayed using
"mouse_plot" [1, 3].

If you make use of code provided here please site [1] and [2].

Refs:
 [1] Andrews (2025), "Exploring Trophic Analysis for Directed Biological Networks"
 [2] MacKay, Johnson & Sansom (2020), "How directed is a directed network", Royal Society Open Science
 [3] Slowinski et al. (2019), "Background EEG connectivity captures the time-course of epileptogenesis in a mouse model of epilepsy", eNeuro

From: Sarah Andrews
'''

################################
###                          ###
###          IMPORTS         ###
###                          ###
################################

import scipy.io
import matplotlib.pyplot as plt
from Tools import trophic_tools as ta
import networkx as nx
import numpy as np
import pandas as pd
from Tools import mouse_plot
import seaborn
import os
import matplotlib as mpl
import random
import copy

random.seed(1)


################################
###                          ###
###        FUNCTIONS         ###
###                          ###
################################

# Transform into a list of weighted edges with items of the form: (from, to, weight)
def make_weighted_edges(network):
    network_length = len(network)
    weighted_edges = []
    for i in range(0, network_length):
        for j in range(0, network_length):
            if network[i][j] != 0:
                weighted_edges.append((i+1, j+1, network[i][j]))
    return weighted_edges


# Load data from one of "all_net_abs.mat", "all_net_pos.mat", or "all_net_neg.mat"
def load_data(file_name):
    os.chdir(r'C:\Users\sarah\OneDrive\Documents\Masters Degree\Research Project\Data and Programs\Data')
    all_data_matlab = scipy.io.loadmat(file_name)
    all_data_unextracted = all_data_matlab[file_name[:-4]][0]
    total_networks = len(all_data_unextracted)
    net_len = len(all_data_unextracted[0][0][0])
    all_data = [all_data_unextracted[i][0] for i in range(0, total_networks)]
    os.chdir(r'C:\Users\sarah\OneDrive\Documents\Masters Degree\Research Project\Data and Programs')
    return (all_data, total_networks, net_len)


# Perform trophic analysis on network
def trophic_analyse(network, make_plot=False, title=None, seed=None,
                    print_info=False, include_labels=False):
    '''
    Perform trophic level analysis on a given network, with optional visualisation.
    INPUTS:
        network : adjacency matrix (list of lists)
        make_plot : bool, optional
            If True, generates a trophic plot using `trophic_tools`.
        title : str, optional
            Title for the plot.
        seed : int, optional
            Random seed for layout reproducibility in the plot.
        print_info : bool, optional
            If True, prints trophic incoherence value.
        include_labels : bool, optional
            If True, includes node labels in the plot and trophic level output.
    OUTPUTS:
        tuple (trophic_levels, incoherence) where:
            trophic_levels : dict
                Mapping of node → trophic level.
            incoherence : float
                Trophic incoherence value of the network.
    '''
    G = nx.DiGraph()
    G.add_weighted_edges_from(make_weighted_edges(network))
    
    # Sort nodes
    H = nx.DiGraph()
    H.add_nodes_from(sorted(G.nodes(data=True)))
    H.add_edges_from(G.edges(data=True))
    G = H
    
    if make_plot:
        plt.rcParams['axes.spines.left'] = True
        plt.rcParams['axes.spines.right'] = True
        plt.rcParams['axes.spines.top'] = True
        plt.rcParams['axes.spines.bottom'] = True
        ta.trophic_plot(G,k=1,title=title,include_labels=True,seed=seed)

    # Get trophic incoherence and levels
    F_0, _ = ta.trophic_incoherence(G)
    tl = ta.trophic_levels(G, include_labels=include_labels)
    
    if print_info:
        print('Trophic incoherence =',round(F_0,3))
    return (tl, F_0)


# Perform virtual resections on a network
def virtual_resections(network, title=None, make_mouse_plot=False, make_plot=False,
                       scale_values_with=[0,0.5,1], absolute=False, give_info=False,
                       with_arrows=False):
    '''
    Perform virtual resections on a network by iteratively removing each node and 
    measuring its effect on trophic coherence.
    INPUTS:
        network : list of lists (adjacency matrix)
            Square adjacency matrix representing network connectivity.
        title : str, optional
            Title for generated plots.
        make_mouse_plot : bool, optional
            If True, generates a spatial "mouse brain" style plot with resections.
        make_plot : bool, optional
            If True, generates a bar-style plot of coherence changes per node.
        scale_values_with : list of float, optional
            Values to use for scaling nodes in the mouse plot [min, mid, max].
        absolute : bool, optional
            If True, uses absolute values of coherence changes when plotting.
        give_info : bool, optional
            If True, returns normalized coherence changes.
        with_arrows : bool, optional
            If True, includes arrows in the mouse plot visualization.
    OUTPUTS:
        dict (if give_info=True)
            Mapping of node index → normalized change in trophic coherence
            after resection.
    '''
    
    net_len = len(network)
    
    # Trophic coherence before node removal
    (_, incoherence) = trophic_analyse(network)
    initial_coherence = 1-incoherence

    # Trophic coherence after node removal, for each node
    removal_coherences = {}
    for i in range(0, net_len):
        # Remove row and column of i
        new_net = []
        for x in range(0, net_len):
            if x!=i:
                row = []
                for y in range(0, net_len):
                    if y!=i:
                        row.append(network[x][y])
                new_net.append(row)

        # Compute coherence for new network
        (_, incoherence) = trophic_analyse(new_net)
        removal_coherences[i+1] = 1-incoherence

    # Normalize
    normalized_changes = {}
    for i in range(1, net_len+1):
        normalized_changes[i] = (removal_coherences[i]-initial_coherence)/initial_coherence

    if make_plot:
        plt.figure(figsize=(8,4))
        mpl.rcParams['font.family'] = 'Arial'
        plt.plot([0,31], [0]*2, 'k-')
        for i in range(1, net_len+1):
            if normalized_changes[i] < 0:
                plt.plot([i,i], [0,normalized_changes[i]], '-', color=(1,0.5,0))
                plt.plot(i, normalized_changes[i], 'o', color=(1,0.5,0))
            else:
                plt.plot([i,i], [0,normalized_changes[i]], '-', color=(0,0.8,0))
                plt.plot(i, normalized_changes[i], 'o', color=(0,0.8,0))
        plt.ylabel("Change in Trophic Coherence")
        plt.xlabel("Resectioned Node")
        plt.xticks([i for i in range(1,31)])
        font = {"fontweight": "bold", "fontsize": 13}
        plt.title(title, font)
        plt.show()            

    if make_mouse_plot:
        mouse_plot.mouse_plot(network, list(normalized_changes.values()), node_size_range=[1, 35],
                              scale_values_with=scale_values_with, title=title,
                              take_absolute_value=absolute, with_arrows=with_arrows)

    if give_info:
        return normalized_changes


def second_diff_virtual_resection(network, day0_network, title=None, make_mouse_plot=False,
                                  scale_values_with=[0,0.5,1], absolute=False, give_info=False,
                                  compute_day_0=True):
    '''
    Perform second-difference virtual resections to compare node removal impact 
    between two networks (e.g., baseline vs. perturbed).
    INPUTS:
        network : list of lists (adjacency matrix)
            Square adjacency matrix for the network under analysis.
        day0_network : list of lists (adjacency matrix)
            Baseline network for comparison.
        title : str, optional
            Title for generated plots.
        make_mouse_plot : bool, optional
            If True, generates a spatial "mouse brain" style plot with second differences.
        scale_values_with : list of float, optional
            Values to use for scaling nodes in the mouse plot [min, mid, max].
        absolute : bool, optional
            If True, uses absolute values of changes when plotting.
        give_info : bool, optional
            If True, returns second-difference changes per node.
        compute_day_0 : bool, optional
            If True, computes resections on the baseline network for comparison.
    OUTPUTS:
        dict (if give_info=True)
            Mapping of node index → difference in normalized coherence change 
            between `network` and `day0_network` resections.
    '''
    net_len = len(network)
    
    # Trophic coherence before node removal
    (_, incoherence) = trophic_analyse(network)
    initial_coherence = 1-incoherence

    if compute_day_0:
        (_, incoherence0) = trophic_analyse(day0_network)
        initial_coherence0 = 1-incoherence0

    # Trophic coherence after node removal, for each node
    removal_coherences = {}
    if compute_day_0:
        removal_coherences0 = {}
        
    for i in range(0, net_len):
        # Remove row and column of i
        new_net = []
        if compute_day_0:
            new_net0 = []
        for x in range(0, net_len):
            if x!=i:
                row = []
                if compute_day_0:
                    row0 = []
                for y in range(0, net_len):
                    if y!=i:
                        row.append(network[x][y])
                        if compute_day_0:
                            row0.append(day0_network[x][y])
                new_net.append(row)
                if compute_day_0:
                    new_net0.append(row0)

        # Compute coherence for new network
        (_, incoherence) = trophic_analyse(new_net)
        removal_coherences[i+1] = 1-incoherence
        if compute_day_0:
            (_, incoherence0) = trophic_analyse(new_net0)
            removal_coherences0[i+1] = 1-incoherence0

    # Normalize
    normalized_changes = {}
    if compute_day_0:
        normalized_changes0 = {}
    for i in range(1, net_len+1):
        normalized_changes[i] = (removal_coherences[i]-initial_coherence)/initial_coherence
        if compute_day_0:
            normalized_changes[i] = (removal_coherences0[i]-initial_coherence0)/initial_coherence0

    # Calculate second differences
    changed_impact = {}
    for i in range(1, net_len+1):
        if compute_day_0:
            changed_impact[i] = normalized_changes[i]-normalized_changes0[i]
        else:
            changed_impact[i] = normalized_changes[i]-day0_network[i-1]

    if make_mouse_plot:
        mouse_plot.mouse_plot(network, list(changed_impact.values()), scale_values_with=scale_values_with,
                              node_size_range=[1, 35], title=title, with_arrows=False,
                              take_absolute_value=absolute)

    if give_info:
        return changed_impact


# Get single graph properties
def get_single_properties(network, measures=None):
    G = nx.DiGraph()
    G.add_nodes_from([i+1 for i in range(0,len(network))])
    G.add_weighted_edges_from(make_weighted_edges(network))
    properties = ta.single_trophic_measures(G, measures=measures)
    return properties


# Produce correlation plots
def correlation(set1,set2,colour="blue",ylab="",xlab="",title="",axis=None,size=30):
    mpl.rcParams['font.family'] = 'Arial'
    if axis == None:
        plt.scatter(set1,set2, s=size, alpha=0.6, edgecolors="k",c=colour)
        b, a = np.polyfit(set1,set2, deg=1)
        xseq = np.linspace(min(set1), max(set1), num=10)
        plt.plot(xseq, a + b * xseq, color="k", lw=2.5)
        plt.ylabel(ylab, fontsize=12, fontweight="bold")
        plt.xlabel(xlab, fontsize=12, fontweight="bold")
        plt.title(title+"CC = "+str(round(np.corrcoef(set1,set2)[0][1],3)), fontweight="bold", fontsize=13)
        plt.show()
    else:
        axis.scatter(set1,set2, s=size, alpha=0.6, edgecolors="k",c=colour)
        b, a = np.polyfit(set1,set2, deg=1)
        xseq = np.linspace(min(set1), max(set1), num=10)
        axis.plot(xseq, a + b * xseq, color="k", lw=2.5)
        axis.set_ylabel(ylab)
        axis.set_xlabel(xlab)
        axis.set_title(title+"CC = "+str(round(np.corrcoef(set1,set2)[0][1],3)))   


# Transpose network to get adjacency matrix
def transpose(network):
    return ((np.matrix(network)).transpose()).tolist()


################################
###                          ###
### TROPHIC ANALYSE NETWORK  ###
###                          ###
################################

# Network type ("abs", "pos", "neg"), and day (0, 7, 28)
net_type = ("abs",0)

if net_type[1] == 0:
    use_network = 25
elif net_type[1] == 7:
    use_network = 26
elif net_type[1] == 28:
    use_network = 27

# Get data
(all_data, total_networks, net_len) = load_data('all_net_'+net_type[0]+'.mat')
net = transpose(all_data[use_network])
  
# Network trophic levels
(_, _) = trophic_analyse(net, make_plot=True,
                         title='Trophic Levels for Median '+net_type[0].upper()+' Network (Day '+str(net_type[1])+')',
                         seed=2, print_info=False)

# Mouse plot for trophic levels
(trophic_levels, incoherence) = trophic_analyse(net, include_labels=True)
values = list(trophic_levels.values())
mouse_plot.mouse_plot(net, values, scale_values_with=[0, 0.325, 0.75],
                      title='Trophic Levels for Median '+net_type[0].upper()+' Network (Day '+str(net_type[1])+')\n',
                      show_imbalance=True, with_arrows=True)



################################
###                          ###
###    ALL NETWORK MEDIANS   ###
###                          ###
################################

(all_data1, total_networks, net_len) = load_data("all_net_abs.mat")
(all_data2, total_networks, net_len) = load_data("all_net_pos.mat")
(all_data3, total_networks, net_len) = load_data("all_net_neg.mat")

median_networks = [[transpose(all_data1[25]),transpose(all_data1[26]),transpose(all_data1[27])],
            [transpose(all_data2[25]),transpose(all_data2[26]),transpose(all_data2[27])],
            [transpose(all_data3[25]),transpose(all_data3[26]),transpose(all_data3[27])]]

networks = [[[transpose(all_data1[i]) for i in range(0,11)],[transpose(all_data1[i]) for i in range(11,17)],
             [transpose(all_data1[i]) for i in range(17,25)]],
            [[transpose(all_data2[i]) for i in range(0,11)],[transpose(all_data2[i]) for i in range(11,17)],
             [transpose(all_data2[i]) for i in range(17,25)]],
            [[transpose(all_data3[i]) for i in range(0,11)],[transpose(all_data3[i]) for i in range(11,17)],
             [transpose(all_data3[i]) for i in range(17,25)]]]

titles = [[None,None,None],[None,None,None],[None,None,None]]

# Find median values of coherences and trophic levels to plot
values = []
for i in range(0,3):
    subvalues = []
    
    for j in range(0,3):
        subsubvalues = []
        coherences = []
        
        for net in networks[i][j]:
            (trophic_levels, incoherence) = trophic_analyse(net, include_labels=True)
            subsubvalues.append(trophic_levels)
            coherences.append(1-incoherence)
        titles[i][j] = "Coherence = "+str(round(np.median(coherences),3))
            
        # Find median trophic level for each node
        medians = []
        for n in range(1,net_len+1):
            per_node = []
            for m in range(0, len(subsubvalues)):
                per_node.append(subsubvalues[m][n])
            medians.append(np.median(per_node))
        subvalues.append(medians)
        
    values.append(subvalues)

# Generate plot
mouse_plot.mouse_subplots(median_networks, values, [3,3],
                          main_title='Median Trophic Levels Across All Networks',
                          sub_titles=titles, show_imbalance=True, scale_values_with=[0, 0.325, 0.75],
                          node_size_range=[2, 24])



################################
###                          ###
###  TROPHIC LEVEL CHANGES   ###
###                          ###
################################

net_type = "abs"
(all_data, total_networks, net_len) = load_data('all_net_'+net_type+'.mat')
networks = [[transpose(all_data[i]) for i in range(0,11)],
            [transpose(all_data[i]) for i in range(11,17)],
            [transpose(all_data[i]) for i in range(17,25)]]

# Find median trophic level changes over days
all_trophic_levels = []
for i in range(0, len(networks)):
    subvalues = []

    # Get a list of all trophic levels
    for net in networks[i]:
        (trophic_levels0, _) = trophic_analyse(net, include_labels=True)
        subvalues.append(trophic_levels0)
    
    medians = []
    for n in range(1, net_len+1):
        per_node = []
        for m in range(0, len(subvalues)):
            per_node.append(subvalues[m][n])
        medians.append(np.median(per_node))
    all_trophic_levels.append(medians)

# Make plot of changes
mpl.rcParams['font.family'] = 'Arial'
plt.figure(figsize=(9, 5))

plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.bottom'] = True

for i in range(0, net_len):
    plt.plot([i+1,i+1], [0,all_trophic_levels[1][i]-all_trophic_levels[0][i]], 'k-')
    plt.plot([i+1,i+1], [all_trophic_levels[1][i]-all_trophic_levels[0][i],all_trophic_levels[2][i]-all_trophic_levels[0][i]], 'k-')
    
plt.plot([i for i in range(1, net_len+1)], [0]*net_len, 'ko', label="Day 0")
plt.plot([i for i in range(1, net_len+1)],
         [all_trophic_levels[1][i]-all_trophic_levels[0][i] for i in range(0,net_len)],
         'o', label="Day 7", color="purple")
plt.plot([i for i in range(1, net_len+1)],
         [all_trophic_levels[2][i]-all_trophic_levels[0][i] for i in range(0,net_len)],
         'o', label="Day 28", color="deeppink")

font = {"fontweight": "bold", "fontsize": 13}
plt.title("Median Changes in Trophic Level per Node ("+str(net_type.upper())+")",font)
plt.ylabel("Change in Trophic Level from Day 0")
plt.xlabel("Node")
plt.xticks([i for i in range(1,31)])
plt.legend()
plt.show()



################################
###                          ###
###    COHERENCE BOXPLOTS    ###
###                          ###
################################

mpl.rcParams['font.family'] = 'Arial'
fig, ax = plt.subplots(1, 3, figsize=(8, 3))
net_types = ["abs", "pos", "neg"]

for i in range(0, 3):
    net_type = net_types[i]
    (all_data, total_networks, net_len) = load_data('all_net_'+net_type+'.mat')

    if net_type == "pos":
        net_type = "max"
    elif net_type == "neg":
        net_type = "min"

    networks1 = [transpose(data) for data in all_data[0:11]]
    networks2 = [transpose(data) for data in all_data[11:17]]
    networks3 = [transpose(data) for data in all_data[17:25]]

    # Get coherence data
    coherences1 = []
    coherences2 = []
    coherences3 = []
    for net in networks1:
        (_, incoherence) = trophic_analyse(net)
        coherences1.append(1-incoherence)
    for net in networks2:
        (_, incoherence) = trophic_analyse(net)
        coherences2.append(1-incoherence)
    for net in networks3:
        (_, incoherence) = trophic_analyse(net)
        coherences3.append(1-incoherence)
    coherences = [coherences1, coherences2, coherences3]

    # Generate boxplots
    ax[i].boxplot(coherences, showfliers=False)
    
    # Add points to plots
    for x in range(0,3):
        for y in coherences[x]:
            x_coord = x+1 -0.1 + random.random()*0.2
            ax[i].plot(x_coord,y,'bo',markersize=2.5)

    ax[i].set_xticks([1,2,3], ["Day 0","Day 7","Day 28"])
    ax[i].set_title(str(net_type.upper()), fontsize=12)
    ax[i].set_ylim(0,0.3)

fig.supylabel("Trophic Coherence")
fig.suptitle("Trophic Coherences over Each Day", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.show()



################################
###                          ###
###  VIRTUAL RESECT NETWORK  ###
###                          ###
################################

net_type = ("abs", 28)
(all_data, total_networks, net_len) = load_data("all_net_"+net_type[0]+".mat")

if net_type[1] == 0:
    use_network = 25
elif net_type[1] == 7:
    use_network = 26
elif net_type[1] == 28:
    use_network = 27
    
net = transpose(all_data[use_network])
virtual_resections(net, title="Median "+str(net_type[0].upper())+" Network (Day "+str(net_type[1])+")",
                   make_plot=True, make_mouse_plot=True, scale_values_with=[0,0.1,0.2],
                   absolute=2, with_arrows=True)



################################
###                          ###
###  MEDIAN RESECT NETWORKS  ###
###                          ###
################################

(all_data1, total_networks, net_len) = load_data("all_net_abs.mat")
(all_data2, total_networks, net_len) = load_data("all_net_pos.mat")
(all_data3, total_networks, net_len) = load_data("all_net_neg.mat")

median_networks = [[transpose(all_data1[25]),transpose(all_data1[26]),transpose(all_data1[27])],
            [transpose(all_data2[25]),transpose(all_data2[26]),transpose(all_data2[27])],
            [transpose(all_data3[25]),transpose(all_data3[26]),transpose(all_data3[27])]]

networks = [[[transpose(all_data1[i]) for i in range(0,11)],[transpose(all_data1[i]) for i in range(11,17)],
             [transpose(all_data1[i]) for i in range(17,25)]],
            [[transpose(all_data2[i]) for i in range(0,11)],[transpose(all_data2[i]) for i in range(11,17)],
             [transpose(all_data2[i]) for i in range(17,25)]],
            [[transpose(all_data3[i]) for i in range(0,11)],[transpose(all_data3[i]) for i in range(11,17)],
             [transpose(all_data3[i]) for i in range(17,25)]]]

# Get virtual resection median results
values = []
for i in range(0,3):
    subvalues = []
    for j in range(0,3):
        coherences = []
        for net in networks[i][j]:
            value = virtual_resections(net, give_info=True)
            coherences.append(value)
            
        # Find median for each node
        medians = []
        for n in range(1, net_len+1):
            per_node = []
            for m in range(0, len(coherences)):
                per_node.append(coherences[m][n])
            medians.append(np.median(per_node))
            
        subvalues.append(medians)
    values.append(subvalues)

# Generate plot
mouse_plot.mouse_subplots(median_networks, values, [3,3],
                          main_title='Median Virtual Resections Across All Networks',
                          scale_values_with=[0,0.05,0.1], take_absolute_value=2, node_size_range=[2,24])



################################
###                          ###
###     RESECTION CHANGES    ###
###                          ###
################################

mpl.rcParams['font.family'] = 'Arial'
net_type = "abs"
(all_data, total_networks, net_len) = load_data('all_net_'+net_type+'.mat')

networks = [[transpose(all_data[i]) for i in range(11,17)],[transpose(all_data[i]) for i in range(17,25)]]
networks0 = [transpose(all_data[i]) for i in range(0,11)]

# Get initial coherences on day 0
coherences = []
for net in networks0:
    value = virtual_resections(net, give_info=True)
    coherences.append(value)
    
# Find median for each node on day 0
values0 = []
for n in range(1, net_len+1):
    per_node = []
    for m in range(0, len(coherences)):
        per_node.append(coherences[m][n])
    values0.append(np.median(per_node))

# Find second difference values
all_second_diffs = []
for i in range(0, len(networks)):
    subvalues = []
    for net in networks[i]:
        changed_impact = second_diff_virtual_resection(net, values0, absolute=True,
                                                       give_info=True, compute_day_0=False)
        subvalues.append(changed_impact)
    medians = []
    for n in range(1, net_len+1):
        per_node = []
        for m in range(0, len(subvalues)):
            per_node.append(subvalues[m][n])
        medians.append(np.median(per_node))
    all_second_diffs.append(medians)

# Generate plot
plt.figure(figsize=(9, 5))

plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.bottom'] = True

for i in range(0, net_len):
    plt.plot([i+1,i+1], [0,all_second_diffs[0][i]], 'k-')
    plt.plot([i+1,i+1], [all_second_diffs[0][i],all_second_diffs[1][i]], 'k-')
plt.plot([i for i in range(1, net_len+1)], [0]*net_len, 'ko', label="Day 0")
plt.plot([i for i in range(1, net_len+1)], [all_second_diffs[0][i] for i in range(0,net_len)],
         'o', label="Day 7", color="purple")
plt.plot([i for i in range(1, net_len+1)], [all_second_diffs[1][i] for i in range(0,net_len)],
         'o', label="Day 28", color="deeppink")
font = {"fontweight": "bold", "fontsize": 13}
plt.title("Median Changes in Trophic Coherence After Virtual Resection (Compared to Day 0) per Node ("+str(net_type.upper())+")",font)
plt.ylabel("Difference Compared to Day 0")
plt.xlabel("Node")
plt.xticks([i for i in range(1,31)])
plt.legend()
plt.show()



################################
###                          ###
###  2ND DIFFERENCE MEDIANS  ###
###                          ###
################################

(all_data1, total_networks, net_len) = load_data("all_net_abs.mat")
(all_data2, total_networks, net_len) = load_data("all_net_pos.mat")
(all_data3, total_networks, net_len) = load_data("all_net_neg.mat")

median_networks = [[transpose(all_data1[26]),transpose(all_data1[27])],
            [transpose(all_data2[26]),transpose(all_data2[27])],
            [transpose(all_data3[26]),transpose(all_data3[27])]]

networks = [[[transpose(all_data1[i]) for i in range(11,17)],
             [transpose(all_data1[i]) for i in range(17,25)]],
            [[transpose(all_data2[i]) for i in range(11,17)],
             [transpose(all_data2[i]) for i in range(17,25)]],
            [[transpose(all_data3[i]) for i in range(11,17)],
             [transpose(all_data3[i]) for i in range(17,25)]]]

networks0 = [[transpose(all_data1[i]) for i in range(0,11)],
             [transpose(all_data2[i]) for i in range(0,11)],
             [transpose(all_data3[i]) for i in range(0,11)]]

titles = [[None,None],[None,None],[None,None]]

# Get median virtual resection results on day 0
values0 = []
for i in range(0,3):
    coherences = []
    for net in networks0[i]:
        value = virtual_resections(net, give_info=True)
        coherences.append(value)
    medians = []
    for n in range(1, net_len+1):
        per_node = []
        for m in range(0, len(coherences)):
            per_node.append(coherences[m][n])
        medians.append(np.median(per_node))
    values0.append(medians)

# Get median 2nd difference results on days 7 and 28
values = [] ; signif = []
for i in range(0,3):
    subvalues = []
    sub_signifs = []
    
    for j in range(0,2):
        signif_decrease = []
        coherences = []
        
        # Calculate significant decreases
        num_with_decrease = {k : 0 for k in range(1, net_len+1)}
        for net in networks[i][j]:
            value = second_diff_virtual_resection(net, values0[i], give_info=True, compute_day_0=False)
            for k in range(1, net_len+1):
                if value[k] < 0:
                    num_with_decrease[k]+= 1
            coherences.append(value)
            
        # Find median for each node
        medians = {}
        for n in range(1, net_len+1):
            per_node = []
            for m in range(0, len(coherences)):
                per_node.append(coherences[m][n])
            medians[n] = np.median(per_node)
            
            # Check if decrease is significant
            if num_with_decrease[n] / len(networks[i][j]) >= 0.8 and np.median(per_node) < -0.05:
                signif_decrease.append(n)

        subvalues.append(list(medians.values()))
        sub_signifs.append(signif_decrease)
        
    values.append(subvalues)
    signif.append(sub_signifs)

# Generate plot
mouse_plot.mouse_subplots(median_networks, values, [3,2],
                          main_title='Median Second Differences Across All Networks',
                          scale_values_with=[0,0.05,0.1], take_absolute_value=1,
                          node_size_range=[2,24], highlight_nodes=signif)



################################
###                          ###
###  STRATEGIC EDGE REMOVAL  ###
###                          ###
################################

net_type = ("abs",0)
if net_type[1] == 0:
    use_network = 25
elif net_type[1] == 7:
    use_network = 26
elif net_type[1] == 28:
    use_network = 27

(all_data, total_networks, net_len) = load_data('all_net_'+net_type[0]+'.mat')
net = transpose(all_data[use_network])

# Total edges to remove
total_removed = 500

# Remove edges in order of weight
num_removed = 0
net1 = copy.deepcopy(net)
networks = [net]

while num_removed < total_removed:
    try:
        listy = [min([xi for xi in x if xi>0]) for x in net1]
        minimum = min([xi for xi in listy if xi>0])
        i = listy.index(minimum) ; j = net1[i].index(minimum)
        num_removed += 1
        net1[i][j] = 0
        networks.append(copy.deepcopy(net1))
    except:
        total_removed = num_removed

# Find network trophic properties
incoherences = [] ; trophic_ranges = [] ; trophic_means = []
for i in range(0, len(networks)):
    props = get_single_properties(networks[i])
    [F_0,Tr,Tm] = [props["incoherence"], props["trophic range"], props["trophic mean"]]
    incoherences.append(F_0) ; trophic_ranges.append(Tr) ; trophic_means.append(Tm)

plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.bottom'] = True

# Generate plot
mpl.rcParams['font.family'] = 'Arial'
fig, axs = plt.subplots(1,3,figsize=(9, 3))

axs[0].plot([i for i in range(0,total_removed+1)], incoherences, 'r-')
axs[0].set_xlabel("Total Edges Removed")
axs[0].set_ylabel("Incoherence")

axs[1].plot([i for i in range(0,total_removed+1)], trophic_ranges, 'b-')
axs[1].set_xlabel("Total Edges Removed")
axs[1].set_ylabel("Trophic Range")

axs[2].plot([i for i in range(0,total_removed+1)], trophic_means, 'g-')
axs[2].set_xlabel("Total Edges Removed")
axs[2].set_ylabel("Trophic Mean")

fig.suptitle("The Impact of Edge Removal on Trophic Measures", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()



################################
###                          ###
###  NORMALITY VS COHERENCE  ###
###                          ###
################################

# Normality Calculater
def find_normality(network):
    # Get eigenvalues of W
    W = np.array(network)
    eigenvalues, _ = np.linalg.eig(W)
    # Sum of squares of absolute values of eigenvalues
    summation1 = 0
    for eig in eigenvalues:
        summation1 += abs(eig)**2
    # Find Frobenius norm of W
    summation2 = 0
    for i in range(0, len(network)):
        for j in range(0, len(network)):
            summation2 += abs(network[i][j])**2
    return (summation1/summation2)

net_types = ["abs","pos","neg"]
all_incoherences = []
all_normalities = []

for net_type in net_types:
    (all_data, total_networks, net_len) = load_data('all_net_'+net_type+'.mat')
    networks = [transpose(data) for data in all_data[0:25]]
    for net in networks:
        (_, incoherence) = trophic_analyse(net)
        all_incoherences.append(incoherence)
        all_normalities.append(find_normality(net))

correlation(all_incoherences,all_normalities,colour="red",size=15,xlab="Trophic Incoherence",
            ylab="Normality",title="The Correlation Between Trophic Coherence and Normality: ")




