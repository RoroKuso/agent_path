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

## First work: Ramanana's model

To compute paths from differents points, Ramanana built a simple model: instead of difficult human with joints and muscles, he uses the "Yoyo man", modeling human walk as a wheel. From there he computes the power needed to walk, from which we can deduce speed and time. He then apply a shortest path algorithm on a map to obtain trajectories and traversal durations.

This project uses some of Ramanana's code, which can be found in `/macro_path_ramanana`

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

## Acknowledgement

- Marie-Paule Cani, supervisor, head VISTA team from LIX lab(Laboratoire d'Informatique de l'école polytechnique).
- Paul Boursin, co-supervisor, phd student at LIX.
- Adrien Ramanana, ex intern at LIX.