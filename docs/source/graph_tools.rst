graph\_tools module
-------------------

This file is used to generate graphs froms height maps. The original height map is `data/maps/LA-CARTE-964.tif`.

Since height maps are 2D grids, we define the distance as the manhattan distance(example: distance from (0, 0) to (2, 3) is d=2+3=5).

Once we have a height map, we have to choose which edges to add. The goal is for an RL-trained agent to cross this map.
We would like to make it possible for the agent not to climb a mountain directly, but to use a longer and smoother path, as human usually do. Since the original map is pretty steep, we have to decide on the possible edges from one point.
In Ramanana's project, this was not an issue as his Yoyo-man could climb any slope. His model allowed to go in all direction to any of the adjacent nodes(d=1 or 2 for diagonals).

We define parameters that allow to go beyond 1 on the x_axis and y_axis. And another for the maximum distance allowed::

    MAX_X = 2
    MAX_Y = 2
    MAX_TOT = 3

This means that from point (0, 0) we can at most, go to point with manhattan distance equal 3.

.. automodule:: graph_tools
   :members:
   :undoc-members:
   :show-inheritance:
