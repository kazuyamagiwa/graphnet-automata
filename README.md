# graphnet-automata
"graphnet-automata" is a study of the evolution of graphs using cellular automaton-like methods, written in Python.  This study was inspired by the [recent announcement](https://writings.stephenwolfram.com/2020/04/finally-we-may-have-a-path-to-the-fundamental-theory-of-physics-and-its-beautiful/) from Wolfram concerning his [project to find the fundamental theory of physics](https://www.wolframphysics.org/).  Although graphnet-automata's approach is not as elegant, it involves the conversion of "seed graphs" to matrices for manipulation using cellular automaton-like methods. 
 
# Premise
 
You can learn how to making cute physics simulations (looks retro game).
 
![](https://cpp-learning.com/wp-content/uploads/2019/05/pyxel-190505-161951.gif)
 
This animation is a "Cat playing on trampoline"!
You can get basic skills for making physics simulations.
 
# Features
 
graphnet-automata uses [NetworkX](https://networkx.github.io/) and Numpy.
```python
import networkx
import numpy
```
# Requirement
 
* Python 3.7
* NetworkX 
* Numpy
* Numba
* python-louvain (for "community" module)
 
graphnet-automata has been tested under [Anaconda for Windows](https://www.anaconda.com/distribution/) and [Google Colaboratory](https://colab.research.google.com/).
 
```bash
conda create -n graphnet-automata pip python=3.7 Anaconda
activate graphnet-automata
```
 
# Installation
 
Install NetworkX with pip command.
 
```bash
pip install networkx
pip install python-louvain
```
 
# Usage
 
Please create python code named "demo.py".
And copy &amp; paste [Day4 tutorial code](https://cpp-learning.com/pyxel_physical_sim4/).
 
Run "demo.py"
 
```bash
python demo.py
```
 
# Note
 
Linux and macOS environments have not yet been tested.
 
# Author
 
* Masakazu Yamagiwa
* Email : myamagiwa@gmail.com
 
# License
 
"Physics_Sim_Py" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
