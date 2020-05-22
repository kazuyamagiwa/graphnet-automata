# graphnet-automata
"graphnet-automata" is a study of the evolution of graphs using cellular automaton-like methods, written in Python.  This study was inspired by the [recent announcement](https://writings.stephenwolfram.com/2020/04/finally-we-may-have-a-path-to-the-fundamental-theory-of-physics-and-its-beautiful/) from Wolfram concerning his [project to find the fundamental theory of physics](https://www.wolframphysics.org/).  Although graphnet-automata's approach is not as elegant, it involves the conversion of "seed graphs" to matrices for manipulation using cellular automaton-like methods. 
 
# Premise
 
Starting from a undirected graph with a few nodes (a "seed graph"), graphnet-automata manipulates its node connections, as represented as a matrix, by first padding the matrix by one layer (adding two new nodes).  Depending on the number of neighboring node connections, the matrix is updated, much like in a cellular automaton.  This is repeated in a recursive fashion (i.e. "evolved"), and the resulting matrix is converted into a graph.  To further characterize the graph, a community detection method is employed to visualize the number of communities that have formed.

For example, a seed graph of three nodes

![](https://github.com/kazuyamagiwa/graphnet-automata/blob/master/images/g1_0.png)
 
 evolves into a graph that is distinctly separated into two communities.
 
![](https://github.com/kazuyamagiwa/graphnet-automata/blob/master/images/g1_100_community.png)

Initial experiments have shown that the evolved graph shows a "bonding" or "anti-bonding" graph, depending on whether the number of nodes in the seed graph are odd or even.  See the [notebook "ga01.ipynb"](https://github.com/kazuyamagiwa/graphnet-automata/blob/master/ga01.ipynb) for details.
 
# Features
 
graphnet-automata uses [NetworkX](https://networkx.github.io/), [NumPy](https://numpy.org/), [Numba](http://numba.pydata.org/) and [python-louvain](https://github.com/taynaud/python-louvain).
```python
import networkx
import numpy
from numba import jit
import community
```
# Requirement
 
* Python 3.6.9
* NetworkX 2.2
* NumPy 1.16.4
* Numba 0.45.1
* python-louvain 0.14 (for "community" library)
 
graphnet-automata has been tested under [Anaconda for Windows](https://www.anaconda.com/distribution/) and [Google Colaboratory](https://colab.research.google.com/).
 
```bash
conda create -n graphnet-automata pip python=3.6.9 Anaconda
activate graphnet-automata
```
 
# Installation
 
Install NetworkX, Numba, and python-louvain with pip command.
 
```bash
pip install networkx
pip install numba (for x86/x86_64 platforms)
pip install python-louvain
```
 
# Usage
 
Please view the Jupyter Notebooks in this repository.  Note that the recursive calculations may take several minutes, depending on your environment.  Work on a python file (.py) is under way.
 
# Note
 
Linux and macOS environments have not yet been tested.
 
# Author
 
* Masakazu Yamagiwa
* Email : myamagiwa@gmail.com
 
# License
 
"graphnet-automata" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
