import numpy as np
import pandas as pd
import networkx as nx
import os

from graph_tools import *
from map_tools import ref_res

def create_graph(n, x, y, tot):
    """Map to graph + saving.
    
    Parameters
    ----------
    n : int
        Discretization parameter.
    x : int
        `MAX_X` parameter.
    y : int
        `MAX_Y` parameter.
    tot : int
        `MAX_TOT` parameter.
    """
    df = pd.read_csv(f"data/maps/height_map_964_N{n}_real.csv")
    grid = df.to_numpy()
    dim = grid.shape[0]
    print(f"shape is {grid.shape}")
    graph = generate_graph(grid, n, x, y, tot)
    save_graph(graph, f"data/graphs/graph_964_N{n}_real_X{x}Y{y}M{tot}.csv")
    
def update_weights(n, x, y, tot, weight_file, update_function, suffix):
    """
    Load and update a graph's weights then save it.
    
    Parameters
    ----------
    n : int
        Discretization parameter.
    x : int
        `MAX_X` parameter.
    y : int
        `MAX_Y` parameter.
    tot : int
        `MAX_TOT` parameter.
    weight_file : str
        Where to find the weights.
    update_function : function
        Function that modify the graph with the correct weights.
    suffix : str
        Suffix added at the end of the graph's name to save the new graph.
    """
    slopes2val = {}
    with open(f"data/weights/{weight_file}", 'r') as f:
        lines = f.readlines()
        slopes = lines[0].split(' ')
        values = lines[1].split(' ')
        for i in range(len(slopes)):
            slopes2val[float(slopes[i])] = float(values[i])
            
    print("Loading graph")
    graph = load_graph(f"data/graphs/graph_964_N{n}_real_X{x}Y{y}M{tot}.csv")
    print("Done.")
    print("Updating weights")
    update_function(graph, slopes2val)
    print("Done.")
    print("Saving graph")
    save_graph(graph, f"data/graphs/graph_964_N{n}_real_X{x}Y{y}M{tot}_{suffix}.csv")
    print("Done.")
    
def update_weight_ramanana(n, x, y, tot):
    """
    Updating a graph with weight as time from Ramanana's article.
    
    Parameters
    ----------
    n : int
        Discretization parameter.
    x : int
        `MAX_X` parameter.
    y : int
        `MAX_Y` parameter.
    tot : int
        `MAX_TOT` parameter.
    """
    ground_dict = {}
    ground_dict[-1] = [1,1, "Default"]
    ground_dict[str(np.array([255, 255, 255]))] = [5e5, 1, "Alluvions"]
    ground_dict[str(np.array([255, 198, 0]))] = [20e9, 10, "Calcaire"]
    ground_dict[str(np.array([51, 200, 35]))] = [10e9, 10, "Marne"]
    ground_dict[str(np.array([213, 29, 29]))] = [5e7, 3, "Eboulis"]
    ground_dict[str(np.array([0, 0, 0]))] = [-1e2, 1, "Eau profonde"]
    ground_dict[str(np.array([0, 216, 255]))] = [2e5, 1, "Eau peu profonde"]
    print("Loading map")
    df = pd.read_csv(f"data/maps/height_map_964_N{n}_real.csv")
    grid = df.to_numpy()
    print("Done")
    print("Loading graph")
    nG = load_graph(f"data/graphs/graph_964_N{n}_real_X{x}Y{y}M{tot}.csv")
    print("Done")
    ground_data = get_ground_data(grid, "data/maps/sol-couleur.png", ground_dict, n)
    print("Computing weights")
    ramanana_time_as_weight(nG, ground_data, n, grid.shape[0])
    print("Done")
    save_graph(nG, f"data/graphs/graph_964_N{n}_real_X{x}Y{y}M{tot}_RamananaWeighted.csv")

    
def generate_paths(n, x, y, tot, suffix):
    """
    Load graph with weights and compute paths, printed over an image.
    
    Parameters
    ----------
    n : int
        Discretization parameter.
    x : int
        `MAX_X` parameter.
    y : int
        `MAX_Y` parameter.
    tot : int
        `MAX_TOT` parameter.
    suffix : str
        Suffix of the graph to load. Defines how its weighted.
    """
    dim = ref_res + n * (ref_res - 1)
    print("Loading graph")
    graph = load_graph(f"data/graphs/graph_964_N{n}_real_X{x}Y{y}M{tot}_{suffix}.csv")
    print("   Done.")
    origin = (467, 184)
    u = (origin[1] * ref_res + origin[0]) * (n + 1)
    end_points = [[154, 377], [446, 272], [241, 581], [598,94], [826, 889]]
    background = "data/maps/topo-light.png"
    print("Computing paths")
    for end_point in end_points:
        print(f"   Starting path {origin} --> {end_point}")
        v = (end_point[1] * ref_res + end_point[0]) * (n + 1)
        fwd_path, fweight_sum = get_path(graph, u, v)
        bwd_path, bweight_sum = get_path(graph, v, u)
        output_dir = f"data/paths/graph_964_N{n}_real_X{x}Y{y}M{tot}_{suffix}/{end_point[0]}_{end_point[1]}"
        os.makedirs(output_dir, exist_ok=True)
        show_path(fwd_path, bwd_path, background, output_dir, n)
        # Create a text file where are written the coordinates of the end point as well as the distance and the duration of the path
        with open(output_dir + "/path_info.txt", 'w') as f:
            f.write(f"End point: {end_point}\n")
            f.write(f"Distance (forward): {path_distance(graph, fwd_path, n, dim)}\n")
            f.write(f"Total weight (forward): {fweight_sum}\n")
            f.write(f"Distance (backward): {path_distance(graph, bwd_path, n, dim)}\n")
            f.write(f"Total weight (backward): {bweight_sum}\n")
        print("   Done.")
    

        
if __name__ == '__main__':
    print()