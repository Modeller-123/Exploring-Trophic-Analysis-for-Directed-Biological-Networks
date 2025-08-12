'''
This Python module is for conducting trophic analysis as introduced by MacKay, Johnson and Sansom in [2], on gene regulatory networks.
The code below makes use of the adapted "trophic_tools" [1, 2], and was written by Sarah Andrews for [1].

If you make use of code provided here please site [1] and [2].

Refs:
 [1] Andrews (2025), "Exploring Trophic Analysis for Directed Biological Networks"
 [2] MacKay, Johnson & Sansom (2020), "How directed is a directed network", Royal Society Open Science

From: Sarah Andrews
'''

################################
###                          ###
###          IMPORTS         ###
###                          ###
################################

import csv
import networkx as nx
from Tools import trophic_tools as ta
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler
import random
import matplotlib as mpl
import numpy as np
import copy

random.seed(1)

################################
###                          ###
###        FUNCTIONS         ###
###                          ###
################################

def make_full_network(net_name, print_info=False):
    '''
    Build a directed regulatory network from a `.network` file.
    INPUTS:
        net_name : (str) Name of the network file (without extension) located in `network/`.
        print_info : (bool, optional) If True, prints number of regulatory genes.
    OUTPUTS:
        list [G, regulators] where:
            G : networkx.DiGraph
                Directed graph with edges weighted by binding strength (>= 0.8).
            regulators : list of str
                Nodes with at least one outgoing edge (regulatory genes).
    '''
    filename = f"network/{net_name}.network"
    G = nx.DiGraph()
    
    with open(filename, 'r', encoding='utf-8') as f:
        next(f)  # Skip header line
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue
            tf_target = parts[0]
            weighted_binding = float(parts[4])

            # Handle different dash encodings in the source-target format
            if 'â€”' in tf_target:
                source, target = tf_target.split('â€”')
            elif '—' in tf_target:
                source, target = tf_target.split('—')
            else:
                source, target = tf_target.split('-')

            # Only keep edges with high binding strength
            if weighted_binding >= 0.8:
                G.add_edge(source.strip(), target.strip(), weight=weighted_binding)
                
    regulators = [u for u, d in G.out_degree() if d > 0]
    if print_info:
        print("\nNumber of regulatory genes:", len(regulators))
        
    return [G, regulators]


def get_connections(net_name, print_info=False):
    '''
    Extract significant edges and regulatory nodes from a differential network file.
    INPUTS:
        net_name : (str) Name of the network (used to locate `influence/infl_<net_name>_neoblast_250k_diffnetwork.tsv`).
        print_info : (bool, optional) If True, prints number of regulatory genes.
    OUTPUTS:
        list [connections, regulatory_nodes] where:
            connections : list of [source, target, weight]
                Edges with weight > 0.8.
            regulatory_nodes : list of str
                Unique source nodes from significant connections.
    '''
    # Read edges from TSV file
    connections = []
    with open("influence/infl_"+net_name+"_neoblast_250k_diffnetwork.tsv") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            connections.append(line)

        # Remove header row
        connections = connections[1:]

        # Keep only high-weight edges
        new_connections = []
        for i in range(0, len(connections)):
            connections[i][2] = float(connections[i][2])
            if connections[i][2] > 0.8:
                new_connections.append(connections[i])
    connections = new_connections

    # Identify unique regulatory nodes (sources)
    regulatory_nodes = list({connection[0] for connection in connections})
            
    if print_info:
        print("\nNumber of regulatory genes:", len(regulatory_nodes))
    
    return [connections, regulatory_nodes]


def largest_connected_component(G):
    '''
    Return the largest weakly connected component of a directed graph.
    INPUTS:
        G : networkx.DiGraph
    OUTPUTS:
        networkx.DiGraph Subgraph corresponding to the largest weakly connected component.
    '''
    components = nx.weakly_connected_components(G)
    largest = max(components, key=len)
    return G.subgraph(largest).copy()


def trophic_analyse(network, make_plot=False, title="", seed=None, print_info=False,
                    remove_percent=0, exclude=[], hide_edges=False, include_range=None,
                    custom_y_scale=True, highlight=None, scale=None, save_not_show=None,
                    axs=None, percent_edges=None, colours=None):
    '''
    Perform trophic level analysis on a given network, with optional visualization.
    INPUTS:
        network : list or networkx.DiGraph. Either a list of [source, target, weight] edges or an existing DiGraph.
        make_plot : bool, optional. If True, generates a trophic plot using `trophic_tools`.
        title : str, optional. Title for the plot.
        seed : int, optional. Random seed for layout reproducibility.
        print_info : bool, optional. If True, prints network statistics and trophic incoherence.
        remove_percent : float, optional. Percentage of non-regulatory nodes to randomly remove before analysis.
        exclude : list of str, optional. Nodes to exclude from removal (e.g., regulatory genes).
        hide_edges : bool, optional. If True, hides edges in the plot.
        include_range : tuple or list, optional. Y-axis range to display in the plot.
        custom_y_scale : bool, optional. Whether to use a custom Y scale in the plot.
        highlight : list of str, optional. Nodes to highlight in a different color.
        scale : dict, optional. Node scaling values (e.g., size) mapped by node name.
        save_not_show : str, optional. If set, saves the plot to the given path instead of showing it.
        axs : matplotlib axis, optional. Axis object for plotting.
        percent_edges : float, optional. Percentage of edges to display (for clarity in dense graphs).
        colours : list, optional. Predefined colors for nodes.
    OUTPUTS:
        tuple (trophic_levels, incoherence, G) where:
            trophic_levels : dict
                Mapping of node → trophic level.
            incoherence : float
                Trophic incoherence value.
            G : networkx.DiGraph
                Processed graph after node removal and filtering.
    '''
    # Convert to directed graph if input is an edge list
    if isinstance(network, nx.DiGraph):
        G = network
    else:
        G = nx.DiGraph()
        G.add_weighted_edges_from(network)

    N = len(G.nodes)
    if print_info:
        print("Number of nodes in original graph:", N)
        print("Number of edges in original graph:", len(G.edges))

    # Random node removal (excluding specified nodes)
    shuffled_nodes = list(G.nodes)
    random.shuffle(shuffled_nodes)
    number_to_remove = int((N - len(exclude)) * remove_percent / 100)
    count = 0
    removed = 0
    while removed < number_to_remove:
        choice = shuffled_nodes[count]
        if choice not in exclude:
            G.remove_node(choice)
            removed += 1
        count += 1
        
    # Keep only the largest weakly connected component
    G = largest_connected_component(G)
    if print_info:
        print(f"Number of nodes after {remove_percent}% removal:", len(G))

    # Optionally normalize node scale values
    if scale is not None:
        new_scale = {node: scale[node] for node in G.nodes}
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(np.array(list(new_scale.values())).reshape(-1, 1)).flatten()
        scale = dict(zip(new_scale.keys(), scaled_values))

    # Visualization
    if make_plot:
        # Create custom colors if highlighting specific nodes
        if highlight is not None:
            colours = [[1, 0, 0] if gene in highlight else [0, 0, 1] for gene in G.nodes]
        
        plt.rcParams['axes.spines.left'] = True
        plt.rcParams['axes.spines.right'] = True
        plt.rcParams['axes.spines.top'] = True
        plt.rcParams['axes.spines.bottom'] = True
        
        ta.trophic_plot(
            G, k=1, title=title, include_labels=False, seed=seed,
            hide_edges=hide_edges, include_range=include_range,
            custom_y_scale=custom_y_scale, colours=colours,
            save_not_show=save_not_show, scale=scale,
            axs=axs, percent_edges=percent_edges
        )

    # Compute trophic measures
    F_0, _ = ta.trophic_incoherence(G)
    tl = ta.trophic_levels(G, include_labels=True)
    if print_info:
        print("Trophic incoherence:", round(F_0, 3))
        
    return (tl, F_0, G)


def correlation(set1, set2, colour="blue", ylab="", xlab="", title="", axis=None, size=30):
    '''
    Create a scatter plot with linear regression line and correlation coefficient.
        INPUTS:
        set1 : list or array-like. X-axis values.
        set2 : list or array-like. Y-axis values.
        colour : str, optional. Color of scatter points.
        ylab : str, optional. Label for Y-axis.
        xlab : str, optional. Label for X-axis.
        title : str, optional. Plot title prefix (correlation coefficient is appended).
        axis : matplotlib axis, optional. If provided, plot on this axis instead of creating a new figure.
        size : int, optional. Marker size for scatter points.
    '''
    mpl.rcParams['font.family'] = 'Arial'
    
    if axis is None:
        fig, axs = plt.subplots(figsize=(6, 4))
        plt.scatter(set1, set2, s=size, alpha=0.6, edgecolors="k", c=colour)
        b, a = np.polyfit(set1, set2, deg=1)
        xseq = np.linspace(min(set1), max(set1), num=10)
        plt.plot(xseq, a + b * xseq, color="k", lw=2.5)
        plt.title(f"{title}: CC = {round(np.corrcoef(set1, set2)[0][1], 3)}",
                  fontweight="bold", fontsize=13)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.tight_layout()
        plt.show()
    else:
        axis.scatter(set1, set2, s=size, alpha=0.6, edgecolors="k", c=colour)
        b, a = np.polyfit(set1, set2, deg=1)
        xseq = np.linspace(min(set1), max(set1), num=10)
        axis.plot(xseq, a + b * xseq, color="k", lw=2.5)
        axis.set_title(f"CC = {round(np.corrcoef(set1, set2)[0][1], 3)}")


###################################
###                             ###
###  SINGLE DIFF NET ANALYSIS   ###
###                             ###
###################################

# Select a network to analyze (net_num index into diff_networks list)
net_num = 8  # In this paper, only "basalgoblet" and "secretory" are plotted

# List of all difference networks
diff_networks = [
    "basalgoblet", "eep", "epidermis", "muscle", "neuron",
    "parenchyma", "phagocytes", "protonephridia", "secretory"
]

# Load high-weight connections and regulatory genes for the chosen network
[connections, regulators] = get_connections(diff_networks[net_num], print_info=True)

# Define trophic level ranges of interest for plotting
ranges_of_interest = [
    [1, 1.065], None, None, None, None, None, None, None, [1.025, 1.055]
]

# Perform trophic analysis:
# - Remove 50% of target nodes (exclude regulatory ones)
# - Plot trophic levels for the remaining network
(tl, F_0, G) = trophic_analyse(
    connections,
    title="Trophic Levels of " + diff_networks[net_num].title() + " Difference Network",
    print_info=True,
    make_plot=True,
    remove_percent=30,
    exclude=regulators,
    hide_edges=False,
    include_range=ranges_of_interest[net_num]
)



###################################
###                             ###
###     HISTOGRAM ANALYSIS      ###
###                             ###
###################################

# Purpose: Compare trophic level distributions (histograms) across networks

diff_networks = [
    "basalgoblet", "eep", "epidermis", "muscle", "neuron",
    "parenchyma", "phagocytes", "protonephridia", "secretory"
]

# Ranges of trophic levels to focus on for each network
ranges_of_interest = [
    [1.008,1.052], [1.055,1.07], [1.002,1.011], [1.029,1.042], [0.998,1.06],
    [1.005,1.04], [1.005,1.02], [1.01,1.03], [1.018,1.04]
]

mpl.rcParams['font.family'] = 'Arial'
fig, ax = plt.subplots(3, 3, figsize=(8, 6))

for net_num in range(0, 9):
    # Load connections and regulators
    [connections, regulators] = get_connections(diff_networks[net_num])

    # Compute trophic levels without node removal
    (tl, _, _) = trophic_analyse(connections)

    # Remove trophic levels of regulatory genes (focus on targets only)
    for gene in regulators:
        tl.pop(gene, None)

    # Remove outliers (top 5% and bottom 1% of trophic levels)
    trophic_levels = np.array(list(tl.values()))
    upper_bound = np.percentile(trophic_levels, 95)
    lower_bound = np.percentile(trophic_levels, 1)
    for gene in list(tl.keys()):
        if not (lower_bound < tl[gene] < upper_bound):
            tl.pop(gene)

    # Plot histogram for the network
    i, j = divmod(net_num, 3)
    ax[i, j].hist(list(tl.values()), bins=500)
    ax[i, j].set_ylim([0, 250])
    ax[i, j].set_title(diff_networks[net_num].title())

# Add shared labels and title
fig.supylabel("Frequency", fontsize=12, fontweight="bold")
fig.supxlabel("Trophic Level", fontsize=12, fontweight="bold")
plt.suptitle(
    "Histograms of Trophic Levels for Target Genes in each Difference Network",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.show()



###################################
###                             ###
###  REGULATORY NODE ANALYSIS   ###
###                             ###
###################################

# Goal: Plot trophic levels for regulatory genes, scaled by number of targets

diff_networks = [
    "basalgoblet", "eep", "epidermis", "muscle", "neuron",
    "parenchyma", "phagocytes", "protonephridia", "secretory"
]
mpl.rcParams['font.family'] = 'Arial'
fig, ax = plt.subplots(3, 3, figsize=(8, 6))

for net_num in range(0, 9):
    # Load connections and regulatory genes
    [connections, regulators] = get_connections(diff_networks[net_num], print_info=True)

    # Initialize scaling dict for regulators (counts of target genes)
    scale = dict.fromkeys(regulators, 0)
    for connection in connections:
        if connection[1] not in regulators:
            scale[connection[0]] += 1

    # Remove all target genes (100% removal), plot remaining regulatory genes
    (tl, _, _) = trophic_analyse(
        connections, print_info=True, make_plot=True,
        exclude=regulators, remove_percent=100, scale=scale,
        title=diff_networks[net_num].title(), seed=4,
        axs=ax[net_num // 3, net_num % 3]
    )

fig.supylabel("Trophic Level", fontsize=12, fontweight="bold")
plt.suptitle(
    "Trophic Levels for Regulatory Genes in each Difference Network",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.show()



###################################
###                             ###
###     COMMON GENE ANALYSIS    ###
###                             ###
###################################

# Purpose: Find genes present in all networks and compare their trophic levels

def subgraphs_with_common_nodes(graph_list):
    '''Return subgraphs containing only nodes common to all graphs.'''
    common_nodes = set(graph_list[0].nodes())
    for G in graph_list[1:]:
        common_nodes &= set(G.nodes())
    return [G.subgraph(common_nodes).copy() for G in graph_list]

diff_networks = [
    "basalgoblet", "eep", "epidermis", "muscle", "neuron",
    "parenchyma", "phagocytes", "protonephridia", "secretory"
]
graphs = []
all_trophic_levels = []

# Collect trophic levels and graphs for all networks
for net_name in diff_networks:
    [connections, regulators] = get_connections(net_name)
    tl, _, G = trophic_analyse(connections, exclude=regulators, remove_percent=100)
    all_trophic_levels.append(tl)
    graphs.append(G)

# Keep only nodes present in every network
subgraphs = subgraphs_with_common_nodes(graphs)

# Filter trophic levels to include only common nodes
new_trophic_levels = [
    {k: all_trophic_levels[i][k] for k in all_trophic_levels[i] if k in subgraphs[i].nodes()}
    for i in range(len(diff_networks))
]

# Compute mean trophic level for each common gene across networks
all_nodes = sorted(set().union(*[sub.nodes() for sub in subgraphs]))
gene_means = {
    gene: np.mean([
        new_trophic_levels[net_idx][gene]
        for net_idx in range(len(subgraphs))
        if gene in new_trophic_levels[net_idx]
    ])
    for gene in all_nodes
}

# Plot deviations from mean trophic level, colored by magnitude
mpl.rcParams['font.family'] = 'Arial'
fig, ax = plt.subplots(figsize=(10, 5))
cmap = plt.cm.coolwarm

for gene in all_nodes:
    x, y = [], []
    mean = gene_means.get(gene, np.nan)
    for net_idx in range(len(subgraphs)):
        if gene in subgraphs[net_idx].nodes():
            value = new_trophic_levels[net_idx].get(gene, np.nan)
            x.append(net_idx)
            y.append(value)
    if x and y:
        deviations = np.array(y) - mean
        max_dev = max(abs(deviations)) if deviations.size > 0 else 1
        norm = Normalize(vmin=-max_dev, vmax=max_dev)
        colors = [cmap(norm(dev)) for dev in deviations]
        plt.scatter(x, y, color=colors, s=50)

plt.xticks(range(len(subgraphs)), [n.title() for n in diff_networks])
plt.xlabel("Cell Type")
plt.ylabel("Trophic Levels")
plt.title(
    "Trophic Levels of Common Regulatory Genes Across Different Networks",
    fontdict={"fontweight": "bold", "fontsize": 13}
)
plt.tight_layout()

# Add colorbar for deviation from mean
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Deviation from Mean Trophic Level')
plt.show()



###################################
###                             ###
###    NODE REMOVAL ANALYSIS    ###
###                             ###
###################################

# Purpose: Measure how removing nodes affects trophic measures

def get_single_properties(network, exclude):
    '''
    Given a network and list of nodes to exclude, keep only excluded nodes
    and compute single-network trophic measures.
    '''
    G = nx.DiGraph()
    G.add_weighted_edges_from(network)
    
    # Remove all target genes (nodes not in `exclude`)
    for node in list(G.nodes):
        if node not in exclude:
            G.remove_node(node)
    
    # Keep only the largest weakly connected component
    largest_cc = largest_connected_component(G)
    
    return ta.single_trophic_measures(largest_cc)

net_num = 8  # Select network to analyze
diff_networks = [
    "basalgoblet", "eep", "epidermis", "muscle", "neuron",
    "parenchyma", "phagocytes", "protonephridia", "secretory"
]

# Load connections and regulators
[connections, regulators] = get_connections(diff_networks[net_num], print_info=True)

# Get trophic levels (excluding regulatory genes, remove 100% of targets)
(tl, _, _) = trophic_analyse(connections, print_info=True, exclude=regulators, remove_percent=100, seed=4)

# Sort genes by trophic level (low → high)
trophic_levels = dict(sorted(tl.items(), key=lambda item: item[1]))
removal_order = list(trophic_levels.keys())
total_removed = 50  # Max nodes to remove

# Stepwise removal: each iteration removes one node from the highest trophic level
num_removed = 0
networks = [connections]
while num_removed < total_removed:
    i = removal_order[-num_removed + 1]  # Select node to remove
    new_connections = [
        c for c in networks[-1]
        if not (c[0] == i or c[1] == i)
    ]
    num_removed += 1
    networks.append(copy.deepcopy(new_connections))

# Compute properties after each removal step
incoherences, trophic_ranges, trophic_means = [], [], []
for net in networks:
    props = get_single_properties(net, regulators)
    incoherences.append(props['incoherence'])
    trophic_ranges.append(props['trophic range'])
    trophic_means.append(props['trophic mean'])

# Plot how measures change as nodes are removed
mpl.rcParams['font.family'] = 'Arial'
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
axs[0].plot(range(total_removed + 1), incoherences, 'r-')
axs[0].set_xlabel("Total Nodes Removed")
axs[0].set_ylabel("Incoherence")

axs[1].plot(range(total_removed + 1), trophic_ranges, 'b-')
axs[1].set_xlabel("Total Nodes Removed")
axs[1].set_ylabel("Trophic Range")

axs[2].plot(range(total_removed + 1), trophic_means, 'g-')
axs[2].set_xlabel("Total Nodes Removed")
axs[2].set_ylabel("Trophic Mean")

fig.suptitle("The Impact of Node Removal on Trophic Measures", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()



###################################
###                             ###
###  SINGLE FULL NET ANALYSIS   ###
###                             ###
###################################

# Purpose: Analyze one complete network (not a difference network)

diff_networks = [
    "basalgoblet", "eep", "epidermis", "muscle", "neoblast",
    "neuron", "parenchyma", "phagocytes", "protonephridia", "secretory"
]

# Select full network for analysis
net_name = diff_networks[4]  # "neoblast"
[G, regulators] = make_full_network(net_name)

# Create subgraph of regulatory genes only
H = G.subgraph(regulators)

# Identify "source" regulatory genes (no incoming edges)
sources = [u for u, d in H.in_degree() if d == 0]

# Run trophic analysis:
# - Remove 100% of targets, keep regulators only
# - Highlight source regulators in red
_, _, _ = trophic_analyse(
    G, exclude=regulators, remove_percent=100, make_plot=True,
    print_info=True, highlight=sources, percent_edges=1,
    title="Trophic Levels for " + net_name.title() + " Regulatory Genes in Full Network"
)



###################################
###                             ###
###    ALL FULL NET ANALYSIS    ###
###                             ###
###################################

# Purpose: Compare regulatory gene trophic levels across multiple full networks
#          relative to the "neoblast" baseline

diff_networks = [
    "basalgoblet", "eep", "epidermis", "muscle", "neoblast",
    "neuron", "parenchyma", "phagocytes", "protonephridia", "secretory"
]

all_trophic_levels = []  # Stores trophic levels per network
all_genes = set()
created_networks = []    # Store graph objects

# Collect trophic levels for each network
for net_name in diff_networks:
    [G, regulators] = make_full_network(net_name, print_info=True)
    tl, _, G = trophic_analyse(G, exclude=regulators, remove_percent=100, print_info=True)
    all_trophic_levels.append(tl)
    all_genes.update(tl.keys())
    created_networks.append(G)

# Prepare color normalization based on deviation from neoblast (index 4)
cmap = plt.get_cmap("coolwarm")
vmax = max(
    abs(tl.get(g, 0) - all_trophic_levels[4].get(g, 0))
    for tl in all_trophic_levels
    for g in tl
)
norm = Normalize(vmin=-vmax, vmax=vmax)

mpl.rcParams['font.family'] = 'Arial'
fig, ax = plt.subplots(3, 3, figsize=(8, 6))

# Plot each network’s regulators colored by deviation from neoblast levels
for i, tl in enumerate(all_trophic_levels):
    if i != 4:  # Skip neoblast itself
        genes = list(tl.keys())
        G = created_networks[i]
        deviations = {g: tl[g] - all_trophic_levels[4][g] for g in genes}

        # Map deviations to colors
        colors = {g: cmap(norm(dev)) for g, dev in deviations.items()}
        new_colors = {g: colors[g] for g in G.nodes}
        colors = new_colors

        k, j = divmod(i if i < 4 else i - 1, 3)
        ta.trophic_plot(
            G, k=1, title=diff_networks[i].title(), include_labels=False,
            hide_edges=True, colours=list(colors.values()), axs=ax[k, j]
        )

fig.supylabel("Trophic Level", fontsize=12, fontweight="bold")
plt.suptitle(
    "Trophic Levels for Regulatory Genes in Full Networks",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.show()



###################################
###                             ###
###  NODE IMBALANCE CORRELATION ###
###                             ###
###################################

# Purpose: Compare trophic level with node imbalance (in-degree minus out-degree)

net_num = 8  # Select network index
diff_networks = [
    "basalgoblet", "eep", "epidermis", "muscle", "neoblast",
    "neuron", "parenchyma", "phagocytes", "protonephridia", "secretory"
]

# Load full network (G) and regulators
[connections, regulators] = make_full_network(diff_networks[net_num])

# Get trophic levels for non-regulatory nodes only
tl, _, G = trophic_analyse(connections, print_info=True, exclude=regulators, remove_percent=100)

# Compute node imbalance for each node (weighted in-degree - weighted out-degree)
imbalances = {
    node: G.in_degree(node, weight='weight') - G.out_degree(node, weight='weight')
    for node in tl.keys()
}

# Create scatter plot of trophic level vs node imbalance
correlation(
    list(tl.values()),
    list(imbalances.values()),
    colour="blue", size=15,
    title="Trophic Level and Node Imbalance",
    ylab="Node Imbalance",
    xlab="Trophic Level"
)



