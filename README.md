# graphnet-automata

Study of graph evolution using cellular automaton-like methods, packaged as a modern [uv](https://docs.astral.sh/uv/) Python project.

![](https://github.com/kazuyamagiwa/graphnet-automata/blob/master/images/ga02_g1_100_community.png)

This study was inspired by the [recent announcement](https://writings.stephenwolfram.com/2020/04/finally-we-may-have-a-path-to-the-fundamental-theory-of-physics-and-its-beautiful/) from Wolfram concerning his [project to find the fundamental theory of physics](https://www.wolframphysics.org/). Although graphnet-automata's approach is not as elegant, it involves the conversion of "seed graphs" to matrices for manipulation using cellular automaton-like methods.

## Premise

Starting from an undirected graph with a few nodes (a "seed graph"), graphnet-automata manipulates its node connections, represented as a matrix, by first padding the matrix with one layer (adding two new nodes). Depending on the number of neighboring node connections, the matrix is updated, much like in a cellular automaton. This is repeated in a recursive fashion (i.e. "evolved"), and the resulting matrix is converted into a graph. To further characterize the graph, a community detection method is employed to visualize the number of communities that have formed.

For example, a seed graph of three nodes

![](https://github.com/kazuyamagiwa/graphnet-automata/blob/master/images/g1_0.png)

evolves into a graph that is distinctly separated into two communities.

![](https://github.com/kazuyamagiwa/graphnet-automata/blob/master/images/g1_100_community.png)

Initial experiments have shown that the evolved graph shows a "bonding" or "anti-bonding" graph, depending on whether the number of nodes in the seed graph are odd or even, like so:

![](https://github.com/kazuyamagiwa/graphnet-automata/blob/master/images/g2_0.png)

evolves into:

![](https://github.com/kazuyamagiwa/graphnet-automata/blob/master/images/g2_100_community.png)

Other novel structures found so far include a whip-like structure:

![](https://github.com/kazuyamagiwa/graphnet-automata/blob/master/images/ga02_g1_100_community.png)

See the [notebook directory](https://github.com/kazuyamagiwa/graphnet-automata/tree/master/notebooks) for details.

## Project layout

```text
.
├── data/                      # Generated HDF5 datasets
├── images/                    # Example figures
├── notebooks/                 # Exploratory Jupyter notebooks
├── src/graphnet_automata/     # Installable package (uv src layout)
│   ├── generator.py           # Shared automaton core
│   ├── degree.py              # Degree-count kernel search
│   ├── generate.py            # Dataset generation
│   ├── optimize.py            # Optuna seed optimization
│   └── search.py              # Entropy-based kernel search
├── tests/
├── pyproject.toml
└── .python-version
```

## Requirements

* Python 3.10+
* Managed via [uv](https://docs.astral.sh/uv/)

Core libraries: NetworkX, NumPy, SciPy, Numba, python-louvain, Matplotlib, h5py, Optuna.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the repository root:

```bash
uv sync
```

For notebooks / optional visualization extras:

```bash
uv sync --extra dev --extra viz
```

## Usage

After `uv sync`, run the CLI entry points:

```bash
uv run graphnet-automata                 # list commands
uv run graphnet-automata-degree          # search by degree-count average
uv run graphnet-automata-generate        # write data/GA_seed_13_0.05_50.h5
uv run graphnet-automata-search          # search by degree-distribution entropy
uv run graphnet-automata-optimize        # Optuna optimization (slow)
```

Or import the shared generator from Python / notebooks:

```python
from graphnet_automata import GeneratorState, kernel_from_index

kernel = kernel_from_index(448)
gen = GeneratorState(nodes=13, prob=0.05, kernel=kernel, steps=100)
adjacency = gen.run()
```

Exploratory work remains in the Jupyter notebooks under `notebooks/`. Recursive calculations may take a minute or more depending on your environment.

## Development

```bash
uv sync --extra dev
uv run pytest
```

## Author

* Masakazu Yamagiwa
* Email: myamagiwa@gmail.com

## License

graphnet-automata is under the [MIT license](https://opensource.org/licenses/MIT).
