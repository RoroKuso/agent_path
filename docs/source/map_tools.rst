map\_tools module
-----------------

This file is used to generate height maps.

For now the only real data we own is a map of small part of Tautavel's valley.
One issue arises: the slopes are too great to be used by an RL-trained agent on a 3D map.

This issue is solved by increasing the number of points on the map. The new heights are estimated as linear means of the values originally present in the map.
But this alone doesn't solve the issue. Indeed, the slopes depend on the choice of edges between points. This is explained in :mod:`graph_tools`.
However, note that increasing the number of points allow for some edges to be less steep, because of how we choose the edges.

.. automodule:: map_tools
   :members:
   :undoc-members:
   :show-inheritance:
