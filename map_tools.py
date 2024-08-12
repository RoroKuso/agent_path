from PIL import Image
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import csv
import pandas as pd 

N = 0
filename = "data/maps/LA-CARTE-964.tif"

def true_height(value: int) -> float:
    return value / 128 * 418

def center_coord(u, v):
    """returns center point of a square with upper left corner `u` and bottom right corner `v`."""
    return (u[0] + (v[0] - u[0]) // 2, 
            u[1] + (v[1] - u[1]) // 2)
    
def linear_mean(arr, N):
    """
    Fill null elements whith weighted average value of left-right elements where left is the closest smaller value divisible by N+1 and right
    is the closest bigger value divisible by N+1 (because the grid is initialiazed this way).
    
    This is a simple approximation, not intended to be perfect.
    """
    m = 0
    dim = len(arr)
    for i in range(dim):
        if i % (N+1) != 0:
            arr[i] = (arr[m] * ((N+1) - (i-m)) + arr[m+(N+1)] * (i-m)) / (N+1)
        else:
            m = i
            
def bilinear_mean(grid, N):
    """
    Same as linear but in 2D. Each null element is replaced by a weighted average of left-right-top-bottom elements.
    """
    dim = grid.shape[0]
    m_i, m_j = 0, 0
    for i in range(dim):
        for j in range(dim):
            if (i % (N+1) != 0) and (j % (N+1) != 0):
                a, b, c, d = grid[i, m_j], grid[m_i, j], grid[m_i + (N+1), j], grid[i, m_j + (N+1)]
                grid[i, j] = (a * ((N+1) - (j - m_j)) + b * ((N+1) - (i - m_i)) + c * (i - m_i) + d * (j - m_j)) / (2 * (N+1))
            else:
                if i % (N+1) == 0:
                    m_i = i
                if j % (N+1) == 0:
                    m_j = j
                
def grid_interpolate(grid: np.ndarray, N):
    """Replace new null values of grid by an approximation of what it could be."""
    dim_x, dim_y = grid.shape
    if (dim_x != dim_y or dim_x < 3):
        print(f"Error in grid_interpolate: grid has wrong shape. is {dim_x} x {dim_y}.")
        exit()
    dim = dim_x
    lines = grid[0:dim:(N+1)]
    columns = grid[:, 0:dim:(N+1)].T
    for line in lines:
        linear_mean(line, N)
    for column in columns:
        linear_mean(column, N)
    
    bilinear_mean(grid, N)
                
            
def start(x):
    global N
    N = x
    
    height_img = Image.open(filename).convert('L')
    pixels_height = true_height(np.array(height_img))
    # pixels_height = (np.arange(9) + 1).reshape(3, 3)

    n = pixels_height.shape[0]
    if n == 1:
        print("Error n = 1")
        exit()
        
    new_n = n + N * (n-1) if n != 2 else N + 2
    new_height_map = np.zeros((new_n, new_n))

    for i in range(new_n):
        for j in range(new_n):
            if (i % (N+1) == 0) and (j % (N+1) == 0):
                row = i // (N+1)
                col = j // (N+1)
                new_height_map[i, j] = pixels_height[row, col]

    grid_interpolate(new_height_map, N)
    new_height_map = np.round(new_height_map, 2)
    df = pd.DataFrame(new_height_map)
    df.to_csv(f"data/maps/height_map_964_N{N}_real.csv", index=False)
    
if __name__ == '__main__':
    start(1)