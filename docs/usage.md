# Usage

## Command-line tools

After `uv sync`, these console scripts are available:

```bash
uv run graphnet-automata                 # list commands
uv run graphnet-automata-degree          # degree-count kernel search
uv run graphnet-automata-generate        # write data/GA_seed_*.h5
uv run graphnet-automata-search          # entropy-based kernel search
uv run graphnet-automata-optimize        # Optuna seed optimization (slow)
```

Recursive evolution over all 512 kernels can take a while depending on
`nodes`, `prob`, and `steps`.

## Python API

```python
from graphnet_automata import GeneratorState, kernel_from_index
import networkx as nx

kernel = kernel_from_index(448)
gen = GeneratorState(nodes=13, prob=0.05, kernel=kernel, steps=100)
adjacency = gen.run()
graph = nx.from_numpy_array(adjacency)
```

## Notebooks

Exploratory notebooks live in `notebooks/` (NetworkX 3+). Frozen NetworkX 2
snapshots are kept under `notebooks/archive/` for historical reference.

## Package layout

```text
src/graphnet_automata/
  generator.py   # shared automaton core
  degree.py      # degree-count search CLI
  generate.py    # HDF5 dataset CLI
  search.py      # entropy search CLI
  optimize.py    # Optuna CLI
```
