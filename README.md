# Intro

The main goal of this project is to compare accurate trajectories of pre historic humans
with computed one.

Considering that pre historic humans had to save energy while going from one place to another, we hope our model and approaches are able to take it into account.

This project use a height map from Tautavel's valley(France). Since we have limited data, we came with a solution for better use of this map.

What is possible:
- Generating more accurate maps from the original one.
- Generating graphs from those maps.
- Applying different weights to the edges and find shortest paths.

See documentation for more details about the algorithm and approximations used in this project.

- [Documentation](#documentation)
- [Requirements](#requirements)
- [First work: Ramanana's model](#first-work-ramananas-model)
- [Conventions](#file-naming-convention)
- [Tutorial](#tutorial)
- [Acknoledgement](#acknoledgements)


## Documentation

The documentation can be generated using Sphinx.

First you need to install the `sphinx` package. Then
```
>> cd docs
>> make html
```

Documentation will then appear in `docs/_build/html/index.html`.

## Requirements

To install necessary packages, use:
```
>> pip install -r requirements.txt
```

## First work: Ramanana's model

To compute paths from differents points, Ramanana built a simple model: instead of difficult human with joints and muscles, he uses the "Yoyo man", modeling human walk as a wheel. From there he computes the power needed to walk, from which we can deduce speed and time. He then apply a shortest path algorithm on a map to obtain trajectories and traversal durations.

This project uses some of Ramanana's code, which can be found in `macro_path_ramanana/`

## File naming convention

This project follows a convention when it comes to naming files(maps, graphs, paths)

- All maps are named `height_map_964_N<n value>_real.csv` where `<n value>` is a positive integer(discretization parameter: used to increase the original map resolution). They are stored in `data/maps/` along `data/maps/LA-CARTE-964.tif`, `data/maps/topo-light.png` and `data/maps/sol-couleur.png` which are respectivly the original height map used for the project, a png of the map and ground data of the map as a color map(each color is matched with a specific ground). Then `964` is simply a reminder that the original maps' resolution is 964x964. Finally, `real` means real heights in meter, not in pixels, otherwise it would be replaced by `pixel`. 

- All graphs are named `graph_964_N<n value>_real_X<x value>Y<y value>M<max value>` or `graph_964_N<n value>_real_X<x value>Y<y value>M<max value>_<suffix>`. The first one holds a graph structure without weights, whereas the second one holds weights values in addition. `<suffix>` is what allows to understand what value is used for the weight.Examples of used suffixes:
    - `RamananaWeighted`: weights are time values as described in Ramanana's paper.
    - `torque_by_dist`: weights are torques values, computed from Paul Boursin's RL-trained agent.
    - `time`: weights are duration to cross the edges, computed from Paul Boursin's RL-trained agent.
    
    Next, `X`, `Y` and `M` values define the edges of the graph. It means that from a node at coordinates (i, j) an edge can land to another node whose first coordinate is in the range `i-X...i+X`, second coordinate in the range `j-Y...j+Y` and the maximum length of the edge is `M`(using manhattan distance). This allows to have edges along less steep paths. Graphs are stored in `data/graphs/`.

- Paths are stored in folder where the name are the same as the graphs used to generate them. The folder contains other folders, one for each path computed.

- Folder `data/weights/` stores values computed by Paul Boursin's agent. For now it's stores values for a range of slopes.


## Tutorial

### Generating a map

Since the original map resolution is 964 x 964, we found a way to increase the resolution without loosing information. The whole purpose of this increase is to add more edge possibilities. In particular, we want less steep slopes. 
The discretization process is simple: let `N` be a positive integer and `map` our reference map(`n x n` with `n = 964`). Then for each 2 adjacents points, we add `N` points between them. The new resolution becomes `dim x dim` where `dim = n + N * (n - 1)`.
Our reference map can be found at `data/maps/LA-CARTE-964.tif`.

To generate a new map, we use code from `map_tools.py`:
```Python
from PIL import Image
import numpy as np

height_img = Image.open("data/maps/LA-CARTE-964.tif").convert('L')
pixels_height = true_height(np.array(height_img))
n = pixels_height.shape[0]
new_n = n + N * (n-1)
new_height_map = np.zeros((new_n, new_n))
for i in range(new_n):
    for j in range(new_n):
        if (i % (N+1) == 0) and (j % (N+1) == 0):
            row = i // (N+1)
            col = j // (N+1)
            new_height_map[i, j] = pixels_height[row, col]

grid_interpolate(new_height_map, N)
```
After increasing the resolution, we fill the new points with new values thanks to `grid_interpolate()`.

To save it, you can do the following:
```Python
import pandas as pd

df = pd.DataFrame(new_height_map)
df.to_csv(f"data/maps/height_map_964_N{N}_real.csv", index=False)
```

### Generating a graph
To create a graph from a map generated with some value of `N`, we use code from `graph_tools.py`:
```Python
import numpy
import pandas as pd

df = pd.read_csv(f"data/maps/height_map_964_N{n}_real.csv")
grid = df.to_numpy()
dim = grid.shape[0]
graph = generate_graph(grid, n, x, y, tot)
save_graph(graph, f"data/graphs/graph_964_N{n}_real_X{x}Y{y}M{tot}.csv")
```
Where `x`, `y` and `tot` are the X Y and M values mentioned earlier.

### Updating the weights
There is no unique solution for this, it depends on what you expect the weights to be.
Nonetheless, if you have weight values to store, do it in `data/weights/`.
For this exemple


## Acknoledgements

- Marie-Paule Cani, supervisor, head VISTA team from LIX lab(Laboratoire d'Informatique de l'école polytechnique).
- Paul Boursin, co-supervisor, phd student at LIX.
- Adrien Ramanana, ex intern at LIX.