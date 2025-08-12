'''
This Python module is for conducting trophic analysis as introduced by MacKay, Johnson and Sansom in [2], and also provides tools developed for network visualisation using these methods.
Adjustments have been made to the original functions provided by Bazil Sansom to suit the data analysed in [1]. Additional functions have also been included.
Equation numbers refered to in annotation are from [2].

If you make use of code provided here please site [1] and [2].

Functions included and adapted from [2] are:
- trophic_levels : returns trophic levels as per [2]
- trophic_incoherence : returns trophic incoherence as per [2] (where trophic coherence is 1-incoherence)
- trophic_layout : returns a layout where y-possitions are given by trophic levels, and x-possitions based on a modified force-directed graph drawing algorithm to spread nodes out on the x-axis. For reproducibility, user can save and specify seed.
- trophic_plot : plots network according to trophic_layout

Original functions from [1] are:
- single_trophic_measures : returns a list of single trophic properties (from one network), such as incoherence, trophic range, and trophic mean 
- double_trophic_measures : returns a list of double trophic properties (for comparing two networks), such as level change and correlaton

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

import numpy as np
import networkx as nx
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl


################################
###                          ###
###    TROPHIC ANALYSIS      ###
###                          ###
################################

def trophic_levels(G, include_labels=False, zero_value=0):
    '''
    This function takes a networkx graph object as input, and calculates trophic levels.
    INPUTS:
        G : networkx graph object
        include_labels : bool, to optionally return dictionary with node labels next to trophic levels
        zero_value : translation of the trophic levels (set the zero_value level (after translation to min=0) to zero)
    OUTPUTS:
        h : array or dictionary of trophic levels
    '''
    
    # Check inputs
    if not isinstance(G, nx.Graph):
        msg = "This function takes networkx graph object."
        raise ValueError(msg)
    elif not isinstance(include_labels, bool):
        msg = "This function takes include_labels as True or False (bool)."
        raise ValueError(msg)
    elif not (isinstance(zero_value, float) or isinstance(zero_value, int)):
        msg = "This function takes zero_value as an int or float."
        raise ValueError(msg)
        
    # Check network weakly connected
    G2 = G.to_undirected(reciprocal=False,as_view=True)
    
    if nx.is_connected(G2):
        W = nx.adjacency_matrix(G)                          # Weighted adjacency matrix as Compressed Sparse Row format sparse matrix of type '<class 'numpy.longlong'>'
        in_weight = np.matrix(W.sum(axis=0)).A1             # Eq.2.1
        out_weight = np.matrix(W.sum(axis=1)).A1            # Eq.2.1
        u = in_weight + out_weight                          # (Total) weight Eq.2.2        
        v = in_weight - out_weight                          # Imbalance Eq.2.3 (the difference between the flow into and out of the node)                
        L = diags(u, 0) - (W + W.transpose())               # (Weighted) graph-Laplacian operator Eq.2.5
        L[0,0 ]= 0
        h = spsolve(L, v)                                   # Solve Eq.2.6 for h using sparse solver spsolve
        h = np.round(h - h.min() - zero_value, decimals=10) # Put lowest node at zero and remove numerical error

        # Optionally include labels and return a dictionary
        if include_labels == True:
            h = dict(zip(G,h))
        return h
    else:
        # Should extend to identify components and obtain trohpic levels for each of these.
        msg = 'Network must be weakly connected.'
        raise ValueError(msg)
        

def trophic_incoherence(G):
    ''' 
    This function takes networkx graph object as input, and returns trophic levels and coherence.
    INPUTS:
      G : networkx graph object
    OUTPUTS:
      F_0 : trophic incoherence
      h : array of trophic levels
    '''
    h = trophic_levels(G)
    W = nx.adjacency_matrix(G)
    hj, hi = np.meshgrid(h, h)
    H = np.power([hj-hi-1],2)
    F_0 = (W.multiply(H)).sum() / W.sum()

    return F_0, h


def single_trophic_measures(G, measures=None):
    ''' 
    This function takes networkx graph object as input, and returns requested single trophic properties.
    INPUTS:
      G : networkx graph object
      measures : list of strings (optional)
          Possible measures : 'incoherence', 'trophic range', 'trophic mean'
    OUTPUTS:
      measurements : list of requested measurements (in order given)
    '''

    # Check inputs
    if measures == None:
        measures = ['incoherence','trophic range','trophic mean']
    elif not(isinstance(measures,list)):
        measures = [measures]

    measures = list(set(measures)) # Remove duplicates
    measurement_dict = dict.fromkeys(measures)
    
    if 'incoherence' in measures:
        F_0,_ = trophic_incoherence(G)
        measurement_dict['incoherence'] = F_0
                
    if ('trophic range' in measures) or ('trophic mean' in measures):
        h = trophic_levels(G)
        Tr = max(h) - min(h) # Range in trophic levels

        # Mean of trophic levels with minimum level forced to zero
        Tm = 0
        minh = min(h)
        for level in h:
            Tm += (level-minh)
        Tm = Tm/len(h)

        # Store in dictionary
        if 'trophic range' in measures:
            measurement_dict['trophic range'] = Tr
        if 'trophic mean' in measures:
            measurement_dict['trophic mean'] = Tm

    return measurement_dict


def double_trophic_measures(G, G_star, measures=None):
    ''' 
    This function takes two networkx graph objects as input, and returns requested double trophic properties.
    INPUTS
      G : networkx graph object
      G_star : networkx graph object (target)
      measures : list of strings (optional)
          Possible measures : 'level change', 'correlation'
    OUTPUTS:
      measurements : list of requested measurements (in order given)
    '''

    # Check inputs
    if measures == None:
        measures = ['level change','correlation']
    elif not(isinstance(measures,list)):
        measures = [measures]

    measures = list(set(measures)) # Remove duplicates
    measurement_dict = dict.fromkeys(measures)
    
    if 'level change' in measures:
        # Find mean of trophic levels
        h = trophic_levels(G) ; h_star = trophic_levels(G_star)
        Tm = 0 ; Tm_star = 0
        minh = min(h) ; minh_star = min(h_star)
        for level in h:
            Tm += (level-minh)
        for level in h_star:
            Tm_star += (level-minh_star)
        Tm = Tm/len(h) ; Tm_star = Tm_star/len(h_star)

        # Translate by setting mean to zero
        h = trophic_levels(G, zero_value=Tm, include_labels=True) ; h_star = trophic_levels(G_star, zero_value=Tm_star, include_labels=True)

        # Get level differences        
        h_nodes = list(h.keys()) ; h_star_nodes = list(h_star.keys())
        L_changes = []
        for node in h_nodes.keys():
            if node in h_star_nodes.keys():
                L_changes.append(abs(h[node] - h_star[node]))
        L_change = np.mean(L_changes)

        measurement_dict['level change'] = L_change

    if 'correlation' in measures:
        h = trophic_levels(G, include_labels=True)
        h_star = trophic_levels(G_star, include_labels=True)
        
        h_nodes = list(h.keys()) ; h_star_nodes = list(h_star.keys())
        h_use = [] ; h_star_use = []
        for node in h_nodes.keys(): # Only use matching nodes
            if node in h_star_nodes.keys():
                h_use.append(h[node])
                h_star_use.append(h_star[node])
        corr = np.corrcoef(h_use, h_star_use)[0, 1]

        measurement_dict['correlation'] = corr

    return measurement_dict


################################
###                          ###
###      VISUALISATION       ###
###                          ###
################################

def trophic_layout(G, 
                   k=None, 
                   ypos=None, 
                   iterations=50, 
                   seed=None,    
                   threshold=1e-4):
    ''' 
    This function position nodes in network G using modified Fruchterman-Reingold layout. 
    The layou spreads the nodes on the x-axis taking y-possitions as given. 
    By default the function uses for y-possitions trophic levels as defined in [2], 
    but may also be passed any y-possitions user chooses to define.
    
    REQUIRED INPUTS:
        G : networkx graph object. Positions will be assigned to every node in G.
    
    OPTIONAL INPUTS:
        k : integer or None (default=None). If None the distance is set to 1/sqrt(nnodes) 
            where nnodes is the number of nodes.  Increase this value to spread nodes farther apart on x-axis .
        ypos : array or None (default=None). Initial y-positions for nodes. If None, then use
            trophic levels as defined by [2]. Alternatively user can pass any desired possitions as ypos. 
        iterations : integer (default=50)
            Maximum number of iterations taken.
        seed : integer, tuple, or None (default=None). For reproducible layouts, specify seed for random number generator.
            This can be an integer; or the output seedState obtained from a previous run of trophic_layout or trophic_plot in order to reporduce
            the same layout result.
            If you run:
               pos1, seedState = trophic_layout(G)
               pos2, _ = trophic_layout(G,seed=seedState)
            then pos1==pos2
        threshold: float (default = 1e-4)
            Threshold for relative error in node position changes.
            The iteration stops if the error is below this threshold.
    
    OUTPUTS:
        pos : a dictionary of possitions for each node in G
        seedState : (tuple) the seedState needed to reproduce layout obtained (e.g. if you run:
                        pos1, seedState = trophic_layout(G)
                        pos2, _ = trophic_layout(G,seed=seedState)
                  then pos2==pos1    
    '''
    
    if seed is None or isinstance(seed,int):
        np.random.seed(seed) # sets seed using default (None) or user specified (int) seed
        seedState = np.random.get_state() # saves seed state to be returned (for reproducibility of result)
    elif isinstance(seed,tuple): # allows user to pass seedState obtained from previous run to reproduce same layout
        np.random.seed(None)
        np.random.set_state(seed)
        seedState=seed
    else:
        msg = '"Seed should be None (default); integer or tuple (use seedState which is output of trophic_layout).'
        raise ValueError(msg)
        
    import networkx as nx
    
    # Check networkx graph object
    if not isinstance(G, nx.Graph):
        msg='This function takes networkx graph object'
        raise ValueError(msg)
    
    # Check network weakly connected
    G2 = G.to_undirected(reciprocal=False,as_view=True)
    if not nx.is_connected(G2):
        msg='Network must be weakly connected'
        raise ValueError(msg)
        
    A = nx.to_numpy_array(G)
    dim=2
    nnodes, _ = A.shape
    
    A=A+np.transpose(A) # symmetrise for layout algorithm

    if ypos is None:
        h=trophic_levels(G)
        pos = np.asarray(np.random.rand(nnodes, dim), dtype=A.dtype)
        pos[:,1]=h
        # random initial positions
        #pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
    else:
        # pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
        pos = np.asarray(np.random.rand(nnodes, dim), dtype=A.dtype)
        pos[:,1]=ypos
        # make sure positions are of same type as matrix
        pos = pos.astype(A.dtype)
        
    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    
    # the initial "temperature"  is about .1 of domain area
    # this is the largest step allowed in the dynamics.
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / float(iterations + 1)
    
    for iteration in range(iterations):
        
        # matrix of difference between points
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
           # This is nnodes*nnodes*2 array giving delta_ij1=x_i-x_j delta_ij2=y_i-y_j
        
        # distance between points 
        distance = np.linalg.norm(delta, axis=-1) 
             # This is the nnodes*nnodes euclidian distance matrix with elements
             # d_ij=sqrt((x_i-x_j)^2+(y_i-y_j)^2)
                
        # enforce minimum distance of 0.01
        np.clip(distance, 0.01, None, out=distance)
        
        # displacement "force"
        displacement = np.einsum('ijk,ij->ik',
                                 delta,
                                 (k * k / distance**2 - A * distance / k))
        
        # update positions
        length = np.linalg.norm(displacement, axis=-1) # returns the Euclidean norm of each row in displacement (this norm is also called the 2-norm, or Euclidean LENGTH).
        length = np.where(length < 0.01, 0.01, length)  # enforce: where length<0.01, replace with 0.01, otherwise keep length
        delta_pos = np.einsum('ij,i->ij', displacement, t / length)
        delta_pos[:,1]=0 
        pos += delta_pos 
        
        # cool temperature
        t -= dt
        
        # check convergence
        err = np.linalg.norm(delta_pos) / nnodes
        if err < threshold:
            break
            
    pos = dict(zip(G, pos))
    return pos, seedState


def trophic_plot(G,
                 k=None, 
                 ypos=None,
                 iterations=50, 
                 seed=None,
                 threshold=1e-4,
                 scale=None,
                 title='',
                 node_color=[[0, 0.451, 0.7412]],
                 include_labels=False,
                 hide_edges=False,
                 include_range=None, custom_y_scale=False,
                 colours=None,
                 axs=None,
                 percent_edges=None,
                 save_not_show=None):
    '''
    This is just a wrapper for trophic_layout that automates some plotting decissions.
    
    REQUIRED INPUTS:
        G : networkx graph object. Positions will be assigned to every node in G.
    
    OPTIONAL INPUTS:
    
    Layout options:
        k : integer or None (default=None). If None the distance is set to 1/sqrt(nnodes) 
            where nnodes is the number of nodes.  Increase this value to spread nodes farther apart on x-axis.
        ypos : array or None (default=None). Initial y-positions for nodes. If None, then use
            trophic levels as defined by [2]. Alternatively user can pass any desired possitions as ypos. 
        iterations : integer (default=50)
            Maximum number of iterations taken.
        seed : integer, tuple, or None (default=None). For reproducible layouts, specify seed for random number generator.
           This can be an integer; or the output seedState obtained from a previous run of trophic_layout or trophic_plot in order to reporduce
           the same layout result.
           If you run:
               pos1, seedState = trophic_layout(G)
               pos2, _ = trophic_layout(G,seed=seedState)
            then pos1==pos2
        threshold : float (default = 1e-4)
            Threshold for relative error in node position changes.
            The iteration stops if the error is below this threshold.
        scale : Change size of dots
    
    Plotting options (a selection of draw_networkx options):
        title : (str) optionally provide a title for the chart
        node_color : color string, or array of floats, (default RGB triplet [0, 0.451, 0.7412])
                     Node color. Can be a single color format string, or a  sequence of colors with
                     the same length as nodelist. If numeric values are specified they will be mapped 
                     to colors using the cmap and vmin,vmax parameters.  See matplotlib.scatter for 
                     more details.
        include_labels : bool (default=False). optionally include labels for nodes
        hide_edges : bool. Optionally hide edges for clearer and faster display
        include_range : a list containing two numbers, representing the y-axis range
            custom_y_scale : bool. When combined with custom_y_scale, include_range gives the range of values to zoom in on
        colours : a list of colours for each node
        axs : possible axis to plot on
        percent_edges : percentage of edges to show (number) if show_edges == True
        save_not_show : save figure rather than display it to a pathway provided (str)
        
    OUTPUTS:
        plots input network G
        seedState : (tuple) the seedState needed to reproduce layout obtained (e.g. if you run:
                        seedState = trophic_plot(G)
                        then this exact plot can be replicated by running
                        _ = trophic_plot(G,seed=seedState)
                  
     '''

    # Zoom in on area from include_range[0] to include_range[1]
    def custom_y_transform(y):
        if y <= include_range[0]:
            return (y/include_range[0]) / 3
        else:
            return ((y - include_range[0]) / (include_range[1]-include_range[0])) * (2/3) + (1/3)

    mpl.rcParams['font.family'] = 'Arial'
    F_0,_ = trophic_incoherence(G)
    pos, seedState = trophic_layout(G,k=k,ypos=ypos,iterations=iterations,threshold=threshold,seed=seed)

    # Scale nodes depending on the existance of labels
    nnodes=G.number_of_nodes()
    scaling=1/nnodes
    if include_labels:
        node_size = scaling*8000
        font_color = (1,1,1)
    else:
        node_size = scaling*2000
        font_color = None

    # Scale and colour nodes according to provided scale if applicable
    if scale != None:
        if axs == None:
            scale = {node: max(100+scale[node]*120, 10) for node in G.nodes()}
        else:
            scale = {node: max(10+scale[node]*12, 1) for node in G.nodes()}
        scale = list(scale.values())
        if colours == None:
            colours = [[0, x/(max(scale)), 1-x/(max(scale)*1.4)] for x in scale]
    else:
        scale = node_size

    # Use custom y axis scale if turned on
    if custom_y_scale and include_range != None:
        for node in pos:
            xx, yy = pos[node]
            pos[node] = (xx, custom_y_transform(yy))
        include_range = [0,1]
        
    # Find percentage of edges to show if show_edges == False
    if hide_edges == False and percent_edges != None:
        # Sort edges by weight (descending)
        edges_with_weights = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
        edges_with_weights.sort(key=lambda x: x[2], reverse=True)

        # Select top %
        top_percent_count = max(1, int(len(edges_with_weights) * percent_edges/100))  # At least 1 edge
        top_edges = edges_with_weights[:top_percent_count]

        # Create a new graph with only top edges
        G_top = nx.DiGraph()
        G_top.add_nodes_from(G.nodes(data=True))
        G_top.add_edges_from([(u, v, {'weight': w}) for u, v, w in top_edges])
        G = G_top
    elif hide_edges == True:
        G_top = nx.DiGraph()
        G_top.add_nodes_from(G.nodes(data=True))
        G = G_top

    # If not given subplot axis
    if axs == None:
        fig, ax = plt.subplots(figsize = (8,6))
        nx.draw_networkx(G,with_labels=include_labels,font_color=font_color,pos=pos,node_size=scale,arrowsize=scaling*150,width=scaling*10,ax=ax,node_color=colours);

        if include_range != None:
            plt.ylim(include_range[0], include_range[1])
            
        limits=plt.axis('on')
        ax.tick_params(left=True, labelleft=True)
        plt.ylabel('Trophic Levels')
        plt.xlabel('Trophic Coherence = ' + "{:.2f}".format(1-F_0))
        font = {"fontweight": "bold", "fontsize": 13}
        ax.set_title(title,font)

        if save_not_show != None:
            plt.savefig(save_not_show+"/"+title+".png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
    else:
        ax = axs
        nx.draw_networkx(G,with_labels=include_labels,font_color=font_color,pos=pos,node_size=scale,arrowsize=scaling*150,width=scaling*10,ax=ax,node_color=colours);

        if include_range != None:
            plt.ylim(include_range[0], include_range[1])
            
        limits=plt.axis('on')
        ax.tick_params(left=True, labelleft=True)
        ax.set_xlabel('Trophic Coherence = ' + "{:.2f}".format(1-F_0))
        ax.set_title(title)
    
    return seedState
