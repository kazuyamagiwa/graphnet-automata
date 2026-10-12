# Try the demo online

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kazuyamagiwa/graphnet-automata/blob/master/notebooks/ga01-odd-even-seed-bonding-communities.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kazuyamagiwa/graphnet-automata/HEAD)
[![Documentation Status](https://readthedocs.org/projects/graphnet-automata/badge/?version=latest)](https://graphnet-automata.readthedocs.io/en/latest/?badge=latest)

- **Colab** — open the starter notebook in Google Colab
- **Binder** — launch a temporary JupyterLab session from this repo
- **Streamlit** — local/web UI over the Python package (see below)

# graphnet-automata

Study of graph evolution using cellular automaton-like methods, packaged as a modern [uv](https://docs.astral.sh/uv/) Python project.

**Documentation:** [https://graphnet-automata.readthedocs.io/](https://graphnet-automata.readthedocs.io/)

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

See the [notebook directory](https://github.com/kazuyamagiwa/graphnet-automata/tree/master/notebooks) for details. Pre-NetworkX-3 snapshots are frozen under [`notebooks/archive/`](https://github.com/kazuyamagiwa/graphnet-automata/tree/master/notebooks/archive).

## Project layout

```text
.
├── app.py                     # Streamlit interactive demo
├── data/                      # Generated HDF5 datasets
├── docs/                      # Sphinx / Read the Docs sources
├── images/                    # Example figures
├── notebooks/                 # Active notebooks (NetworkX 3+)
│   └── archive/               # Frozen NetworkX 2-era snapshots
├── src/graphnet_automata/     # Installable package (uv src layout)
│   ├── generator.py           # Shared topological CA core
│   ├── bayesian.py            # NumPy Bayesian CA step (DAG / CMI)
│   ├── bayesian_synth.py      # Synthetic DAG harness (N=10 / 50 / 100)
│   ├── jax_core.py            # JAX @jit port of bayesian.step
│   ├── benchmark.py           # JAX CA vs gCastle PC / GOLEM (default N=100)
│   ├── degree.py              # Degree-count kernel search
│   ├── generate.py            # Dataset generation
│   ├── optimize.py            # Optuna seed optimization
│   └── search.py              # Entropy-based kernel search
├── requirements.txt           # Streamlit Cloud / Binder install list
├── tests/
├── pyproject.toml
└── .python-version
```

## Bayesian automata (Phases 1–6)

On branch `experimental/bayesian-automata`, adjacency evolves by **Bayesian network structure learning** (linear-Gaussian partial correlation / CMI) with elevation-enforced DAGs — not Conway-style topological CA rules.

| Phase | Achievement |
| --- | --- |
| 1 | Batched NumPy `step(A, E, Cov)` — gravity, death, birth |
| 2–4 | Synthetic N=10 recovery; V-structure gravity well; Cov-ranked top-k `Z` |
| 5 | N=50 sparse scale harness + timing |
| 6 | Pure functional JAX `@jit` port (`jax_core`) |

**Key metrics (seeded harness):**

| Setting | Directed F1 | Skeleton |
| --- | --- | --- |
| N=10, k=3 | ~0.81 (multi-seed); seed-0 ~0.94 | F1 ~0.89 |
| N=50 | ~0.39 | F1 ~0.73, **recall ~0.97** |

**GPU (user-reported):** Tesla T4 — 100 ticks in **0.43s** (~**232** ticks/sec), parity with NumPy recovery metrics.

### Install JAX

```bash
uv sync --extra jax                 # CPU jax/jaxlib (portable default)
# Apple Silicon (Metal/MPS):  install jax-mps (or current Apple wheel) separately
# NVIDIA / Colab T4:          see "Colab T4 / CUDA" below — do not rely on --extra jax alone
```

```bash
uv run python -c "from graphnet_automata.jax_core import warmup; print(warmup())"
```

### Colab T4 / CUDA

`uv` extras stay CPU-friendly (`--extra jax`, `--extra gcastle`). On a Colab **T4** (or other NVIDIA GPU), install accelerator wheels separately:

```bash
# Runtime → Change runtime type → T4 GPU
pip install -U "jax[cuda13]" gcastle torch
# If the CUDA extra name differs for your JAX version, follow:
#   https://docs.jax.dev/en/latest/installation.html
```

Then from the repo (or after `pip install -e .`):

```bash
python -m graphnet_automata.benchmark --n 100
```

The benchmark prints device checks at startup (`torch.cuda.is_available()`, `jax.default_backend()`, `jax.devices()`). For **GOLEM** it passes `device_type='gpu'` only when CUDA is available; otherwise it falls back to `device_type='cpu'` with a printed note (gCastle raises `ValueError` if you request GPU without CUDA). Default GOLEM budget is **`num_iter=20000`** (GOLEM-EV: `lambda_1=0.02`, `lambda_2=5.0`, `equal_variances=True`, `learning_rate=1e-3`, `graph_thres=0.3`); the Details line prints the full kwargs. **gCastle GOLEM has no early-stopping / tolerance parameters** — only a fixed Adam loop plus `checkpoint_iter` logging. On CPU-only hosts use `--smoke` to verify import/init without a full `GOLEM.learn`.

### Run the synth harness

```bash
uv run python -m graphnet_automata.bayesian_synth --n 10
uv run python -m graphnet_automata.bayesian_synth --n 10 --seeds 10
uv run python -m graphnet_automata.bayesian_synth --n 50
uv run python -m graphnet_automata.bayesian_synth --n 50 --backend jax
uv run python -m graphnet_automata.bayesian_synth --n 100
```

Reports directed/skeleton TP·FP·FN, F1, and ticks/sec. NumPy remains the default backend. For `n >= 100`, the SEM targets roughly **100–150** true edges.

### Benchmark vs gCastle (PC / GOLEM)

Default **N=100** (same SEM helpers as the synth harness; ~100–150 true edges). Compares skeleton/directed F1 and wall time for **PC** (constraint baseline), **GOLEM** (gradient, GPU-capable), and **Bayesian CA (JAX)**. Prints JAX/PyTorch device checks first.

```bash
uv sync --extra jax --extra gcastle   # CPU wheels; gcastle pulls torch
# or: pip install gcastle
uv run python -m graphnet_automata.benchmark
uv run python -m graphnet_automata.benchmark --n 100 --seed 0 --ticks 100
uv run python -m graphnet_automata.benchmark --n 50   # smaller scale
uv run python -m graphnet_automata.benchmark --golem-iters 20000  # default
uv run python -m graphnet_automata.benchmark --smoke  # CPU: import/init only
```

## Requirements

* Python 3.10+
* Managed via [uv](https://docs.astral.sh/uv/)

Core libraries: NetworkX, NumPy, SciPy, Numba, python-louvain, Matplotlib, h5py, Optuna. Optional: JAX (`--extra jax` or an accelerator wheel), gCastle (`--extra gcastle` or `pip install gcastle`).

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the repository root:

```bash
uv sync
```

Optional extras:

```bash
uv sync --extra dev --extra viz   # notebooks / pyvis
uv sync --extra app               # Streamlit demo
uv sync --extra docs              # Sphinx documentation
uv sync --extra jax               # CPU JAX for bayesian CA (`jax_core`)
uv sync --extra gcastle           # PC / GOLEM benchmark (+ torch)
```

## Streamlit demo

Run the interactive UI locally:

```bash
uv sync --extra app
uv run streamlit run app.py
```

To publish on [Streamlit Community Cloud](https://streamlit.io/cloud): sign in with GitHub, pick this repository, set the main file to `app.py`, and deploy (uses `requirements.txt`).

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

## Documentation

Hosted docs: [https://graphnet-automata.readthedocs.io/](https://graphnet-automata.readthedocs.io/)

Sphinx sources live in `docs/`. Build locally:

```bash
uv sync --extra docs
uv run sphinx-build -b html docs docs/_build/html
```

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
