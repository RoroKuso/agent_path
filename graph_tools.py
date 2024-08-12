
"""
This file is used to generate graphs from height maps.

The original height map is LA-CARTE-964.tif.

We then discretize it mor by adding N nodes between each point.

Final step is adding the edges. Edges have a maximum manhattan distance.
"""

import pandas as pd
import numpy as np
import networkx as nx
import csv
import os
from PIL import Image

# Ramanana's code imports
import macro_path_ramanana.physics_tools as ptools
import macro_path_ramanana.graph as gtools

MAX_X = 4
MAX_Y = 4
MAX_TOT = 5
MAX_TIME = 9e7
dim = 1
N = 0

def get_pixel_in_meter():
    return 4.25 / (N + 1)

def real_plain_distance(i, j, k, l):
    return np.sqrt(((i - k) * get_pixel_in_meter())**2 + ((j - l) * get_pixel_in_meter())**2)

def true_height(value: int) -> float:
    return value / 128 * 418

def index_to_node(i, j):
    return i * dim + j

def node_to_index(u):
    return (u // dim, u % dim)

def heights_to_slope(a, b):
    rad = np.arctan(b / a)
    ret = rad * 180 / np.pi
    return ret

def generate_edges():
    """Returns a list of all possible edges shape as a list of vectors."""
    ret = []
    for x in range(MAX_X + 1):
        for y in range(MAX_Y + 1):
            if (x == y and x > 1) or (x == 0 and y > 1) or (y == 0 and x > 1):
                continue
            if 0 < x + y <= MAX_TOT:
                ret.append((x, y))
    return ret
            
        
def add_edges(graph: nx.MultiDiGraph, edges, u):
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
    """
    i, j = node_to_index(u)
    for (x, y) in edges:
        if (i + x < dim) and (j + y < dim):
            v = index_to_node(i + x, j + y)
            z_length = graph.nodes[u]['height'] - graph.nodes[v]['height']
            length = np.sqrt((x * get_pixel_in_meter())**2 + (y * get_pixel_in_meter())**2 + z_length**2)
            graph.add_edge(u, v, length=length)
            graph.add_edge(v, u, length=length)
        if (i + x < dim) and (0 < j - y):
            v = index_to_node(i + x, j - y)
            z_length = graph.nodes[u]['height'] - graph.nodes[v]['height']
            length = np.sqrt((x * get_pixel_in_meter())**2 + (y * get_pixel_in_meter())**2 + z_length**2)
            graph.add_edge(v, u, length=length)
            graph.add_edge(u, v, length=length)
        
def generate_graph(grid):
    """
    Initizialize a graph from a height map.
    
    Parameters
    ----------
    grid: np.ndarray
        2D Height map.
    """
    graph = nx.DiGraph()
    N_tot = dim * dim
    edges = generate_edges()
    for u in range(N_tot):
        graph.add_node(u, height=grid[*node_to_index(u)])
    for u in range(N_tot):
        if u % 10000 == 0:
            print(f"step u = {u} out of {N_tot}")
        add_edges(graph, edges, u)
        # _add_edges_from(graph, grid, u)
    return graph


def binary_search(l, x):
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
    

def time_as_weight(graph: nx.DiGraph, slopes: dict):
    """
    Updates the edges of `graph`. Weights are updated according to the slope of the edge.
    
    For now, let's consider that `slopes` contains time per meter for different slopes.
    """
    targets = sorted(list(slopes.keys()))
    heights = nx.get_node_attributes(graph, 'height')
    for (u, v, d) in graph.edges.data():
        graph.edges.data()
        h1 = heights[u]
        h2 = heights[v]
        angle = heights_to_slope(h1, h2)
        i = binary_search(targets, angle)
        graph[u][v]['weight'] = slopes[targets[i]] * d['length']
        
def ramanana_time_as_weight(graph: nx.DiGraph, ground):
    """
    Updates the weight(as time) of `graph` using Ramanana's paper. It needs data about the ground to compute dissipated power.
    
    Parameters
    ----------
    graph: nx.Digraph
        The graph to update.
    ground: np.ndarray
        A 2D grid containing ground data for every node.
    """
    nodes_height = nx.get_node_attributes(graph, 'height')
    for (u, v, _) in graph.edges.data():
        i, j = node_to_index(u)
        k, l = node_to_index(v)
        step_dist = 1 # Arbitrary constant in Ramanana's code, distance reached for 1 human step.
        plain_dist = real_plain_distance(i, j, k, l) # Real distance in meter from a 2D pov.
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
          
def get_nearest(i, j):
    """
    Returns the nearest point from the original map (before adding N points between each point).
    """
    if (i % (N+1) == 0) and (j % (N+1) == 0):
        return (i, j)
    else:
        lnode = (i, j - j % (N+1))
        rnode = (i, j + (N+1) - j % (N+1))
        tnode = (i - i % (N+1), j)
        bnode = (i + (N+1) - i % (N+1), j)
        tmp = [lnode, rnode, tnode, bnode]
        d = np.array([real_plain_distance(i, j, *lnode), real_plain_distance(i, j, *rnode), real_plain_distance(i, j, *tnode), real_plain_distance(i, j, *bnode)])
        ind = np.argmin(d)
        return tmp[ind]

def get_ground_data(grid, ground_file, ground_dict):
    dim = grid.shape[0]
    ground_data = np.empty((dim, dim, 2), dtype=np.float32)
    ground_img = Image.open(ground_file)
    pixels_ground = np.array(ground_img)
    for i in range(dim):
        for j in range(dim):
            ref = get_nearest(i, j)
            data = gtools.pixel_to_ground_data(ground_dict, pixels_ground[ref[0], ref[1]])
            ground_data[i, j][0] = data[0]
            ground_data[i, j][1] = data[1]
    return ground_data
    
def save_graph(G: nx.DiGraph, filename):
    """Save a graph in csv format."""
    with open(f"{filename}", "w") as f:
        writer = csv.writer(f, delimiter=' ')
        for (u, h) in nx.get_node_attributes(G, 'height', default=-1).items():
            if h == -1:
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
            else:
                graph.add_edge(int(row[0]), int(row[1]), weight=float(row[2]), length=float(row[3]))
    return graph

def get_path(graph, start, end):
    """
    Compute the shortest path from `start` to `end` and path traversal duration.
    """
    p = nx.shortest_path(graph, start, end, weight='weight')
    t = 0
    for i in range(len(p) - 1):
        u, v = p[i], p[i+1]
        weight = graph[u][v]['weight']
        t += weight
    return p, t

def get_path_ramanana(graph, start, end):
    """
    Compute the shortest path from `start` to `end` and path traversal duration.
    
    Ramanana version: The distance used between 2 nodes is the distance from a 2D point of view.
    """
    p = nx.shortest_path(graph, start, end, weight='weight')
    t = 0
    for i in range(len(p) - 1):
        u, v = p[i], p[i+1]
        weight = graph[u][v]['weight']
        dist = real_plain_distance(*node_to_index(u), *node_to_index(v))
        if dist / weight <= ptools.v_precautious:
            weight = dist / ptools.v_precautious
        t += weight
    return p, t

def test1():
    G = nx.DiGraph(length=0.0, weight=0)
    G.add_edge(0, 1, weight = 3, length = 9)
    G.add_edge(1, 0, weight = -3, length = 9)
    G.add_edge(0, 2, weight = 6, length = 10)
    G.add_edge(2, 0, weight = -6, length = 10)
    G.add_node(0, height=1)
    G.add_node(1, height=2)
    G.add_node(2, height=3)
    
    save_graph(G, "data/graphs/test1.csv")
    nG = load_graph("data/graphs/test1.csv")
    print(nx.get_node_attributes(nG, 'height'))
    print(nG.edges.data())

def main1(n, x, y, tot):
    """Map to graph test."""
    global N
    global dim
    global MAX_X
    global MAX_Y
    global MAX_TOT
    N = n
    MAX_X = x
    MAX_Y = y
    MAX_TOT = tot
    df = pd.read_csv(f"data/maps/height_map_964_N{N}_real.csv")
    grid = df.to_numpy()
    dim = grid.shape[0]
    print(f"shape is {grid.shape}")
    graph = generate_graph(grid)
    # _save_graph(graph, "height_map_964_N0_real.json")
    save_graph(graph, f"data/graphs/graph_964_N{n}_real_X{x}Y{y}M{tot}.csv")

def main2(n, x, y, tot):
    """Loading graph test."""
    nG = load_graph(f"data/graphs/graph_964_N{n}_real_X{x}Y{y}M{tot}.csv")
    print(len(nx.get_edge_attributes(nG, 'length')))
    print(nx.get_edge_attributes(nG, 'length')[(500, 501)])
    
def ramanana1(n, x, y, tot):
    """Updating a graph with weight as time from Ramanana's article."""
    ground_dict = {}
    ground_dict[-1] = [1,1, "Default"]
    ground_dict[str(np.array([255, 255, 255]))] = [5e5, 1, "Alluvions"]
    ground_dict[str(np.array([255, 198, 0]))] = [20e9, 10, "Calcaire"]
    ground_dict[str(np.array([51, 200, 35]))] = [10e9, 10, "Marne"]
    ground_dict[str(np.array([213, 29, 29]))] = [5e7, 3, "Eboulis"]
    ground_dict[str(np.array([0, 0, 0]))] = [-1e2, 1, "Eau profonde"]
    ground_dict[str(np.array([0, 216, 255]))] = [2e5, 1, "Eau peu profonde"]
    global N
    global dim
    N = n
    print("Loading map")
    df = pd.read_csv(f"data/maps/height_map_964_N{n}_real.csv")
    grid = df.to_numpy()
    print("Done")
    dim = grid.shape[0]
    print("Loading graph")
    nG = load_graph(f"data/graphs/graph_964_N{n}_real_X{x}Y{y}M{tot}.csv")
    print("Done")
    ground_data = get_ground_data(grid, "data/maps/sol-couleur.png", ground_dict)
    print("Computing weights")
    ramanana_time_as_weight(nG, ground_data)
    print("Done")
    save_graph(nG, f"data/graphs/graph_964_N{n}_real_X{x}Y{y}M{tot}_RamananaWeighted.csv")
        
    
def ramanana2(graph_file):
    """
    Computing backward and forward paths for different end points.
    """
    print("Loading graph.")
    graph = load_graph(f"data/graphs/{graph_file}")
    weights = nx.get_edge_attributes(graph, 'weight')
    print("Done.")
    origin_id = 184 * ptools.cols + 467
    end_points = [[154, 377], [446, 272], [241, 581], [598,94], [826, 889]]
    backgrounds = ["data/maps/LA-CARTE-964.tif", "data/maps/topo-light.png"]
    for end_point in end_points:
        target_id = end_point[1] * ptools.cols + end_point[0]
        print("forward path")
        fwd_path_packed, fwd_duration = get_path_ramanana(graph, origin_id, target_id)
        print("Done\n Backard path")
        bwd_path_packed, bwd_duration = get_path_ramanana(graph, target_id, origin_id)
        print("Done")
        output_dir = f"data/paths/{graph_file}/{end_point[0]}_{end_point[1]}"
        os.makedirs(output_dir, exist_ok=True) # Create the directory if it does not exist
        gtools.generate_path_on_background(backgrounds, fwd_path_packed, bwd_path_packed, 240, output_dir, weights)
        # Create a text file where are written the coordinates of the end point as well as the distance and the duration of the path
        with open(output_dir + "/path_info.txt", 'w') as f:
            f.write(f"End point: {end_point}\n")
            f.write(f"Distance (forward): {gtools.compute_path_distance(fwd_path_packed)}\n")
            f.write(f"Duration (forward): {fwd_duration}\n")
            f.write(f"Distance (backward): {gtools.compute_path_distance(bwd_path_packed)}\n")
            f.write(f"Duration (backward): {bwd_duration}\n")
        
if __name__ == '__main__':
    # test1()
    # main1(0, 2, 2, 3)
    # main2(0, 2, 2, 3)
    # ramanana1(0, 2, 2, 3)
    ramanana2("graph_964_N0_real_X2Y2M3_RamananaWeighted.csv")
    