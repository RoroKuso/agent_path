import pandas as pd
import numpy as np
import networkx as nx
import json
import csv


MAX_X = 4
MAX_Y = 4
MAX_TOT = 5
dim = 1
N = 0

def get_pixel_in_meter():
    return 4.25 / (N + 1)

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
    for (u, v, d) in graph.edges.data():
        graph.edges.data()
        h1 = graph.nodes[u]['height']
        h2 = graph.nodes[v]['height']
        angle = heights_to_slope(h1, h2)
        i = binary_search(targets, angle)
        graph[u][v]['weight'] = slopes[targets[i]] * d['length']
    
def save_graph(G, filename):
    """Save a graph in csv format."""
    with open(filename, "w") as f:
        writer = csv.writer(f, delimiter=' ')
        for (u, v, d) in G.edges.data():
            writer.writerow([u, v, d['weight'] if 'weight' in d else 0, d['length'] if 'length' in d else 0])

def load_graph(filename):
    """load a graph from csv file."""
    graph: nx.DiGraph = nx.DiGraph()
    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            graph.add_edge(int(row[0]), int(row[1]), weight=float(row[2]), length=float(row[3]))
    return graph


def lol1():
    global N
    global dim
    N = 0
    df = pd.read_csv("height_map_964_N0_real.csv")
    grid = df.to_numpy()
    dim = grid.shape[0]
    print(f"shape is {grid.shape}")
    graph = generate_graph(grid)
    # _save_graph(graph, "height_map_964_N0_real.json")
    save_graph(graph, "graph_964_N0_real.csv")

def lol2():
    nG = load_graph("graph_964_N0_real.csv")
        
if __name__ == '__main__':
    lol2()
    
    