"""File containing graph tools"""

from .physics_tools import *
from PIL import Image
from typing import Dict, Tuple
import networkx as nx
import os

"""
The value of a node is a tuple with the following values: [height, ground_data, vegetation]
The edges hold the time it takes to go from one node to another
"""

def pixel_to_ground_data(ground_dict: Dict[str, Tuple[float, float, str]], pixel: Tuple[int, int, int]) -> Tuple[float, float]:
    """
    Returns the ground caracteristics of a pixel, according to it's color.
    
    Parameters
    ----------
    ground_dict : Dict
        Dict containing ground caracteristics for each type of ground, represented as a color in RGB format.
    pixel : Tuple[int, int, int]
        Color of the pixel
    """
    key = str(pixel)
    if key in ground_dict.keys():
        return np.array(ground_dict[key][:-1])
    else:
        return np.array(ground_dict[-1][:-1])

def add_edge(graph: nx.DiGraph, node_s: Tuple, node_t: Tuple):
    """Adds an edge to the graph.
    
    Parameters
    ----------
    graph: nx.DiGraph
        graph to add edge to.
    node_s: Tuple[int, Tuple[float, Tuple[float, float]]]
        Source node's id and ground data.
    node_t: Tuple[int, Tuple[float, Tuple[float, float]]]
        Target node's id and ground data.
    """
    node_s_id = node_s[0]
    node_t_id = node_t[0]
    node_s_data = node_s[1]
    node_t_data = node_t[1]
    step_dist = 1
    diag = is_diag(node_s_id, node_t_id, cols)
    pixel_dist = pixel_in_meter * np.sqrt(2) if diag else pixel_in_meter
    
    delta_z = (node_t_data[0] - node_s_data[0]) * step_dist / pixel_dist

    # Compare Young Modulus
    if node_s_data[1][0] > node_t_data[1][0]:
        ground_data = node_t_data[1]
    else:
        ground_data = node_s_data[1]
    
    v = compute_walking_speed(delta_z, pixel_dist, ground_data)

    if v == 0:
        time = np.Infinity
    else:
        time = pixel_dist / v

    graph.add_edge(node_s_id, node_t_id, weight=time)

def node_to_row_col(node_id: int, cols: int) -> Tuple[int, int]:
    """returns the (x,y) coordinates of a pixel using it's id in the graph."""
    row = node_id // cols
    col = node_id % cols
    return (row, col)

def is_diag(u: int, v: int, cols) -> bool:
    """Returns true if the edge between adjacent nodes `u` and `v` is a diagonal.
    """
    u_i, u_j = node_to_row_col(u, cols)
    v_i, v_j = node_to_row_col(v, cols)
    if abs(u_i - v_i) > 1 or abs(u_j - v_j) > 1:
        print(f"Error: nodes {u} and {v} are not adjacent")
        return False
    return (abs(u_i - v_i) == 1) and (abs(u_j - v_j) == 1)
    
def generate_graph(height_file: str, ground_file: str, ground_dict: dict) -> nx.DiGraph:
    """Returns a new graph form height and ground data, updating `ground_dict` along the way."""
    # Load image and convert to grayscale
    height_img = Image.open(height_file).convert('L')
    pixels_height = np.array(height_img)
    # pixels_height = pixels_height.astype(np.int32)
    
    ground_img = Image.open(ground_file)
    pixels_ground = np.array(ground_img)
    # pixels_ground = pixels_ground.astype(np.int32)

    # Create graph
    graph = nx.DiGraph()
    rows, cols = pixels_height.shape # could be any other "pixels" image
    for row in range(rows):
        for col in range(cols):
            node_id = row * cols + col
            graph.add_node(node_id, value=(true_height(pixels_height[row, col], 128), pixel_to_ground_data(ground_dict, pixels_ground[row, col])))
    print("finished adding all the nodes")
    
    graph_values = nx.get_node_attributes(graph, 'value')
    for row in range(rows):
        for col in range(cols):
            node_id = row * cols + col
            # Add edges to neighboring nodes
            if col > 0: # left 
                neighbor = node_id - 1
                add_edge(graph, (node_id, graph_values[node_id]), (neighbor, graph_values[neighbor]))
                if row > 0: # bottom left diag
                    neighbor = node_id - cols - 1
                    add_edge(graph, (node_id, graph_values[node_id]), (neighbor, graph_values[neighbor]))
                if row < rows - 1: # top left diaf
                    neighbor = node_id + cols - 1
                    add_edge(graph, (node_id, graph_values[node_id]), (neighbor, graph_values[neighbor]))

            if row > 0: # bottom
                neighbor = node_id - cols
                add_edge(graph, (node_id, graph_values[node_id]), (neighbor, graph_values[neighbor]))
                if col < cols - 1: # bottom right diag
                    neighbor = node_id - cols + 1
                    add_edge(graph, (node_id, graph_values[node_id]), (neighbor, graph_values[neighbor]))

            if col < cols - 1: # right
                neighbor = node_id + 1
                add_edge(graph, (node_id, graph_values[node_id]), (neighbor, graph_values[neighbor]))
                if row < rows - 1: # top right diag
                    neighbor = node_id + cols + 1
                    add_edge(graph, (node_id, graph_values[node_id]), (neighbor, graph_values[neighbor]))
                    
            if row < rows - 1: # top
                neighbor = node_id + cols
                add_edge(graph, (node_id, graph_values[node_id]), (neighbor, graph_values[neighbor]))
                 
    print("finished adding the neighbors")
    return graph


def get_path(graph: nx.DiGraph, start: int, end: int) -> Tuple:
    """Returns shortest path from `start` to `end` in `graph`.
    """
    shortest_path = nx.shortest_path(graph, start, end, weight='weight')
    path_duration = 0
    suspicious_nodes = []
    precautious_nodes = []
    for i in range(len(shortest_path[:-1])):
        u, v = shortest_path[i], shortest_path[i+1]
        edge = (u,v)
        weight = graph[u][v]['weight']
        pixel_dist = pixel_in_meter * np.sqrt(2) if is_diag(u, v, cols) else pixel_in_meter
        # pixel_dist = real_plain_distance()
        if pixel_dist / weight <= v_precautious:
            if pixel_dist / weight <= v_min:
                suspicious_nodes.append(u)
                suspicious_nodes.append(v)
            else:
                precautious_nodes.append(u)
                precautious_nodes.append(v)
            # Set the edge weight to the time it would take to walk at the precautious speed
            weight = pixel_dist / v_precautious
            graph.add_edge(u, v, weight=weight)
        path_duration += weight
        
    return (shortest_path, suspicious_nodes, precautious_nodes), path_duration

def node_to_index(u, dim):
    return (u // dim, u % dim)

def path_viz(background, path, interval, weights, output_path=None, color=[255, 0, 0]):
    """Print the path to a png map and save it."""
    # if it is not already in RGB, transform the image to RGB by copying the first channel 3 times
    # path, suspicious_nodes, precautious_nodes = path
    radius = 1
    if len(background.shape) == 2:
        background = np.stack((background, background, background), axis=2)
    rows, cols, _ = background.shape

    curr_segment_duration = 0
    path_matrix = np.zeros((rows, cols, 3))
    for i, node_id in enumerate(path[:-1]):
        row, col = node_to_index(node_id, cols)
        path_color = color
        duration = weights[(node_id, path[i+1])]
        curr_segment_duration += duration
        for j in range(-radius, radius + 1):
            for k in range(-radius, radius + 1):
                if 0 <= row + j < rows and 0 <= col + k < cols:
                    if j**2 + k**2 <= radius**2:
                        path_matrix[row + j, col + k] = path_color

        # # If the current segment is longer than the interval, color all the pixels around the current node
        # checkpoint_radius = 3 * radius
        # if curr_segment_duration >= interval:
        #     curr_segment_duration = 0
        #     # If the current node is not suspicious or precutious, color the pixels around it with the right color:
        #     if True: # node_id not in suspicious_nodes or True: # and node_id not in precautious_nodes:
        #         for j in range(-checkpoint_radius, checkpoint_radius + 1):
        #             for k in range(- checkpoint_radius, checkpoint_radius + 1):
        #                 if 0 <= row + j < rows and 0 <= col + k < cols:
        #                     if j**2 + k**2 <= checkpoint_radius**2:
        #                         path_matrix[row + j, col + k] = path_color

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
        
def load_image(path):
    img = Image.open(path).convert('RGB')
    # if it is not already in RGB, transform the image to RGB by copying the first channel 3 times
    if len(img.getbands()) == 1:
        img = np.stack((img, img, img), axis=2)
    # if the image is RGBA make t RGB
    img = np.array(img).astype(np.uint32)
    return img

def compute_path_distance(path):
    length = 0
    for i in range(len(path[:-1])):
        # Get the coordinates of two successive points using node_to_row_col:
        row1, col1 = node_to_row_col(path[i], cols)
        row2, col2 = node_to_row_col(path[i+1],cols)
        # Compute the distance between the two points:
        distance = pixel_in_meter * np.sqrt((row1 - row2)**2 + (col1 - col2)**2)
        length += distance
    return length


def generate_path_on_background(backgrounds, fwd_path, bwd_path, interval, output_dir, weights, N=0):
    """Generate pngs containing a map and paths."""
    for background in backgrounds:
        bg = load_image(background)
        dim = bg.shape[0] + N * (bg.shape[0]-1)
        new_bg = np.zeros((dim, dim, 3))
        for i in range(dim):
            for j in range(dim):
                row = i // (N+1)
                col = j // (N+1)
                new_bg[i, j] = bg[row, col]
        ospath = os.path.join(output_dir, os.path.basename(background).split('.')[0] + '.png')
        path_viz(bg, bwd_path, interval, weights, ospath, color=[0, 0, 255])
        bg = load_image(ospath)
        path_viz(bg, fwd_path, interval, weights, ospath, color=[255, 0, 0])


if __name__ == "__main__":
    # @MYLENE: C'est ici qu'il faudra faire la correspondance avec les différents sols.
    # 1 sol = 1 indice (entre les crochets en dessous)
    # La valeur correspond à [Module de Young, Profondeur déformée, Profondeur déformable, Nom]

    ground_dict = {}
    ground_dict[-1] = [1,1, "Default"]
    # entre 10 et 30 GPa pour le calcaire,
    ground_dict[str(np.array([255, 255, 255]))] = [5e5, 1, "Alluvions"]
    ground_dict[str(np.array([255, 198, 0]))] = [20e9, 10, "Calcaire"]
    # entre 1 et 15 GPa pour la marne,
    ground_dict[str(np.array([51, 200, 35]))] = [10e9, 10, "Marne"]
    ground_dict[str(np.array([213, 29, 29]))] = [5e7, 3, "Eboulis"]
    ground_dict[str(np.array([0, 0, 0]))] = [-1e2, 1, "Eau profonde"] # MADE NEGATIVE TO FORCE THE WATER TO BE AVOIDED
    ground_dict[str(np.array([0, 216, 255]))] = [2e5, 1, "Eau peu profonde"]
