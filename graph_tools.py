import numpy as np
import networkx as nx
import csv
import os
from PIL import Image

# Ramanana's code imports
import macro_path_ramanana.physics_tools as ptools
import macro_path_ramanana.graph as gtools


MAX_TIME = 9e+14

def get_pixel_in_meter(N) -> float:
    """Distance between 2 adjacent(up, down, left or right) nodes according to the discretization parameter `N`."""
    return 4.25 / (N + 1)

def real_plain_distance(i, j, k, l, N) -> float:
    """Distance between nodes nodes at `(i, j)` and `(k, l)` without considering height according to the discretization parameter `N`"""
    return np.sqrt(((i - k) * get_pixel_in_meter(N))**2 + ((j - l) * get_pixel_in_meter(N))**2)

def true_height(value: int) -> float:
    """Converts height as pixel to meters"""
    return value / 128 * 418

def index_to_node(i, j, dim) -> int:
    """Converts 2D coordinates from a `dim x dim` grid to a node id."""
    return i * dim + j

def node_to_index(u, dim) -> int:
    """Converts a node id to 2D coordinates in a `dim x dim` grid."""
    return (u // dim, u % dim)

def heights_to_slope(a, b, d) -> float:
    """
    Computes the angle of a slope.
    
    Parameters
    ----------
    a : float
        Height of the from node.
    b : float
        Height of the to node.
    d : float
        Distance in meter between the nodes.
    """
    rad = np.arcsin((b-a)/d)
    ret = rad * 180 / np.pi
    return ret

def generate_edges(MAX_X, MAX_Y, MAX_TOT):
    """Returns a list of all possible edges shape as a list of vectors.
    
    Parameters
    ----------
    MAX_X : int
        Maximum pixel distance along x axis.
    MAX_Y : int
        Maximum pixel distance along y axis.
    MAX_TOT : int
        Maximum length of the edge(using manhattan distance).
    """
    ret = []
    for x in range(MAX_X + 1):
        for y in range(MAX_Y + 1):
            if (x == y and x > 1) or (x == 0 and y > 1) or (y == 0 and x > 1):
                continue
            if 0 < x + y <= MAX_TOT:
                ret.append((x, y))
    return ret
            
        
def add_edges(graph: nx.MultiDiGraph, edges, u, N, dim):
    """
    Add all possible edges from `u` to `graph`, using known possible edges.
    
    Parameters
    ----------
    graph: nx.Digraph
        Graph to add edge to.
    edges: List[Tuple]
        Edges shape (x incr, y incr).
    u: int
        Starting node.
    N : int
        Discretization parameters.
    dim : int
        Size of the map.
    """
    i, j = node_to_index(u, dim)
    for (x, y) in edges:
        if (i + x < dim) and (j + y < dim):
            v = index_to_node(i + x, j + y, dim)
            z_length = graph.nodes[u]['height'] - graph.nodes[v]['height']
            length = np.sqrt((x * get_pixel_in_meter(N))**2 + (y * get_pixel_in_meter(N))**2 + z_length**2)
            graph.add_edge(u, v, length=length)
            graph.add_edge(v, u, length=length)
        if (i + x < dim) and (0 < j - y):
            v = index_to_node(i + x, j - y, dim)
            z_length = graph.nodes[u]['height'] - graph.nodes[v]['height']
            length = np.sqrt((x * get_pixel_in_meter(N))**2 + (y * get_pixel_in_meter(N))**2 + z_length**2)
            graph.add_edge(v, u, length=length)
            graph.add_edge(u, v, length=length)
        
def generate_graph(grid, N, x, y, m):
    """
    Initizialize a graph from a height map.
    
    Parameters
    ----------
    grid: np.ndarray
        2D Height map.
    N : int
        Discretization parameter.
    x : int
        Maximum pixel distance along x axis.
    y : int
        Maximum pixel distance along y axis.
    m : int
        Maximum length of the edge(using manhattan distance).
    """
    graph = nx.DiGraph()
    dim = grid.shape[0]
    N_tot = dim * dim
    edges = generate_edges(x, y, m)
    for u in range(N_tot):
        graph.add_node(u, height=grid[*node_to_index(u, dim)])
    for u in range(N_tot):
        if u % 10000 == 0:
            print(f"step u = {u} out of {N_tot}")
        add_edges(graph, edges, u, N, dim)
    return graph


def binary_search(l, x) -> int:
    """Find the element of `l` that is the closest to `x`."""
    n = len(l)
    if n == 0:
        return -1
    if n == 1:
        return 0
    i, j = 0, n-1
    while abs(i - j) > 1:
        m = (i + j) // 2
        if x < l[m]:
            j = m
        else:
            i = m
    if abs(x - l[i]) < abs(x - l[j]):
        return i
    else:
        return j
    

def time_as_weight(graph: nx.DiGraph, slopes2val: dict):
    """
    Updates the edges of `graph`. Weights are updated according to the slope of the edge.
    
    Parameters
    ----------
    graph : networkx.DiGraph
        Graph to update.
    slopes2val : Dict
        Contains average speed when going along a path of different slopes.
    """
    targets = sorted(list(slopes2val.keys()))
    heights = nx.get_node_attributes(graph, 'height')
    for (u, v, d) in graph.edges.data():
        h1 = heights[u]
        h2 = heights[v]
        angle = heights_to_slope(h1, h2, d['length'])
        if abs(angle) > 30:
            graph[u][v]['weight'] = MAX_TIME
        else:
            i = binary_search(targets, angle)
            graph[u][v]['weight'] =  d['length'] / slopes2val[targets[i]]
        
def torque_per_meter_as_weight(graph: nx.DiGraph, slopes2val: dict):
    """
    Updates the weight of the edges with torque / distance computed from Paul RL agent.
    
    Parameters
    ----------
    graph : networkx.DiGraph
        Graph to update.
    slopes2val : Dict
        Contains the torque per meter value for each angle between 2 points.
    """
    targets = sorted(list(slopes2val.keys()))
    heights = nx.get_node_attributes(graph, 'height')
    for (u, v, d) in graph.edges.data():
        h1 = heights[u]
        h2 = heights[v]
        angle = heights_to_slope(h1, h2, d['length'])
        if abs(angle) > 30:
            graph[u][v]['weight'] = MAX_TIME
        else:
            i = binary_search(targets, angle)
            graph[u][v]['weight'] = slopes2val[targets[i]] * d['length']
        
def ramanana_time_as_weight(graph: nx.DiGraph, ground, N, dim):
    """
    Updates the weight(as time) of `graph` using Ramanana's paper. It needs data about the ground to compute dissipated power.
    
    Parameters
    ----------
    graph: nx.Digraph
        The graph to update.
    ground: np.ndarray
        A 2D grid containing ground data for every node.
    N : int
        Discretization parameters.
    dim : int
        Size of the map.
    """
    nodes_height = nx.get_node_attributes(graph, 'height')
    for (u, v, _) in graph.edges.data():
        i, j = node_to_index(u, dim)
        k, l = node_to_index(v, dim)
        step_dist = 1 # Arbitrary constant in Ramanana's code, distance reached for 1 human step.
        plain_dist = real_plain_distance(i, j, k, l, N) # Real distance in meter from a 2D pov.
        delta_z = (nodes_height[u] - nodes_height[v]) * step_dist / plain_dist
        
        # Compare Young modulus
        if ground[i, j][0] > ground[k, l][0]:
            ground_data = ground[k, l]
        else:
            ground_data = ground[i, j]
        
        speed = ptools.compute_walking_speed(delta_z, plain_dist, ground_data)
        
        if speed == 0:
            time = MAX_TIME
        else:
            time = plain_dist / speed
            
        graph[u][v]['weight'] = time
          
def get_nearest(i, j, N):
    """
    Returns the nearest point from the original map (before adding `N` points between each point).
    
    Parameters
    ----------
    i : int
        First axis coordinate.
    j : int
        Second axis coordinate.
    N : int
        Discretization parameter.
    """
    if (i % (N+1) == 0) and (j % (N+1) == 0):
        return (i // (N+1), j // (N+1))
    else:
        lnode = (i, j - j % (N+1))
        rnode = (i, j + (N+1) - j % (N+1))
        tnode = (i - i % (N+1), j)
        bnode = (i + (N+1) - i % (N+1), j)
        tmp = [lnode, rnode, tnode, bnode]
        d = np.array([real_plain_distance(i, j, *lnode, N), real_plain_distance(i, j, *rnode, N), real_plain_distance(i, j, *tnode, N), real_plain_distance(i, j, *bnode, N)])
        ind = np.argmin(d)
        return (tmp[ind][0] // (N+1), tmp[ind][1] // (N+1))

def get_ground_data(grid, ground_file, ground_dict, N):
    """
    Returns ground data for all nodes.
    
    Paramaters
    ----------
    grid : np.ndarray
        Height map
    ground_file : str
        Stores a color map, classifying each node.
    ground_dict : dict
        Stores data for each ground class.
    N : int
        Discretization parameter.
    """
    dim = grid.shape[0]
    ground_data = np.empty((dim, dim, 2), dtype=np.float32)
    ground_img = Image.open(ground_file)
    pixels_ground = np.array(ground_img)
    for i in range(dim):
        for j in range(dim):
            ref = get_nearest(i, j, N)
            data = gtools.pixel_to_ground_data(ground_dict, pixels_ground[ref[0], ref[1]])
            ground_data[i, j][0] = data[0]
            ground_data[i, j][1] = data[1]
    return ground_data
    
def save_graph(G: nx.DiGraph, filename):
    """Save a graph in csv format."""
    with open(f"{filename}", "w") as f:
        writer = csv.writer(f, delimiter=' ')
        for (u, h) in nx.get_node_attributes(G, 'height').items():
            if h == None:
                print(f"Error in save_graph(): missing height for node {u}")
                exit()
            writer.writerow([u, round(h, 2)])
        for (u, v, d) in G.edges.data():
            if 'length' not in d:
                print(f"Error in save_graph(): missing length for edge ({u}, {v})")
                exit()
            writer.writerow([u, v, round(d['weight'], 2) if 'weight' in d else 0, round(d['length'], 2)])

def load_graph(filename):
    """load a graph from csv file."""
    graph: nx.DiGraph = nx.DiGraph()
    with open(f"{filename}", "r") as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            if len(row) == 2:
                graph.add_node(int(row[0]), height=float(row[1]))
            elif len(row) == 4:
                graph.add_edge(int(row[0]), int(row[1]), weight=float(row[2]), length=float(row[3]))
    return graph

def get_path(graph, start, end):
    """
    Compute the shortest path from `start` to `end` and path traversal sum of weights.
    """
    p = nx.shortest_path(graph, start, end, weight='weight')
    ret = 0
    for i in range(len(p) - 1):
        u, v = p[i], p[i+1]
        weight = graph[u][v]['weight']
        ret += weight
    return p, ret

def get_path_ramanana(graph, start, end, N, dim):
    """
    Compute the shortest path from `start` to `end` and path traversal duration.
    
    Ramanana version: The distance used between 2 nodes is the distance from a 2D point of view.
    """
    p = nx.shortest_path(graph, start, end, weight='weight')
    t = 0
    for i in range(len(p) - 1):
        u, v = p[i], p[i+1]
        weight = graph[u][v]['weight']
        dist = real_plain_distance(*node_to_index(u, dim), *node_to_index(v, dim), N)
        if dist / weight <= ptools.v_precautious:
            weight = dist / ptools.v_precautious
        t += weight
    return p, t

def path_distance(graph, path, N, dim) -> float:
    """Computes the total length of a path with format [u1, u2, u3, ...], the nodes in order of traversal."""
    tot = 0
    heights = nx.get_node_attributes(graph, 'height')
    for t in range(len(path)-1):
        u = path[t]
        v = path[t+1]
        i, j = node_to_index(u, dim)
        k, l = node_to_index(v, dim)
        x, y, z = (i - k) * get_pixel_in_meter(N), (j - l) * get_pixel_in_meter(N), (heights[u] - heights[v])
        tot += np.sqrt(x**2 + y**2 + z**2)
    return tot
        
def load_image(path):
    """Load an image as a RGB array."""
    img = Image.open(path).convert('RGB')
    # if it is not already in RGB, transform the image to RGB by copying the first channel 3 times
    if len(img.getbands()) == 1:
        img = np.stack((img, img, img), axis=2)
    # if the image is RGBA make t RGB
    img = np.array(img).astype(np.uint32)
    return img

def path_viz(background, path, output_path=None, color=[255, 0, 0]):
    """Print the path to a png map and save it.
    
    Parameters
    ----------
    background : numpy.ndarray
        RGB array for the background.
    path : list
        A list of successive nodes.
    """
    radius = 1
    rows, cols, _ = background.shape
    path_matrix = np.zeros((rows, cols, 3))
    for i, node_id in enumerate(path[:-1]):
        row, col = node_to_index(node_id)
        path_color = color
        path_matrix[row, col] = path_color

        # Mark the origin and the end of the path by a big square:
        if i == 0 or i == len(path) - 2:
            for j in range(-radius * 4, radius * 4 + 1):
                for k in range(-radius * 4, radius * 4 + 1):
                    if 0 <= row + j < rows and 0 <= col + k < cols:
                        path_matrix[row + j, col + k] = color

    # Replace the background with the path where the path_img is not black
    path_img = np.any(path_matrix != [0, 0, 0], axis=-1)
    result = np.where(path_img[..., None], path_matrix, background)

    img = Image.fromarray(result.astype(np.uint8))
    if output_path is not None:
        img.save(output_path)

def show_path(fwd_path, bwd_path, background, output_dir, N):
    """Saves paths in a file, printed over a background"""
    bg = load_image(background)
    dim = bg.shape[0] + N * (bg.shape[0]-1)
    new_bg = np.zeros((dim, dim, 3))
    for i in range(dim):
        for j in range(dim):
            row = i // (N+1)
            col = j // (N+1)
            new_bg[i, j] = bg[row, col]
    ospath = os.path.join(output_dir, os.path.basename(background).split('.')[0] + '.png')
    path_viz(new_bg, bwd_path, ospath, color=[0, 0, 255])
    new_bg = load_image(ospath)
    path_viz(new_bg, fwd_path, ospath, color=[255, 0, 0])