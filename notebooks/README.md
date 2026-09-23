# Notebooks

Active exploratory notebooks (NetworkX 3+). Order prefixes match the original
`ga01` / `ga02` / `ga03` sequence; suffixes describe each notebook’s focus.

| File | Contents |
| --- | --- |
| `ga01-odd-even-seed-bonding-communities.ipynb` | Original Numba evolution; 3- vs 4-node seeds; community bonding / anti-bonding |
| `ga01.1-medium-sparse-seed-evolution.ipynb` | Same workflow with medium sparse seeds (13/14 nodes, \(p=0.1\)) |
| `ga01.2-medium-dense-seed-evolution.ipynb` | Medium dense seeds (13/14 nodes, \(p=0.8\)) |
| `ga01.3-large-dense-seed-evolution.ipynb` | Large dense seeds (53/54 nodes, \(p=0.8\)) |
| `ga02-directed-rule-whip-structure.ipynb` | Directed-capable rule; whip-like structure; geodesic analysis |
| `ga03-convolution-kernel-graph-discovery.ipynb` | Convolution-kernel evolution for discovering new graphs |
| `ga03.1-mobius-strip-degree-histogram.ipynb` | Möbius-strip-like ordered graph; degree histogram |
| `ga03.2-interactive-pyvis-visualization.ipynb` | Interactive PyVis exploration of an evolved graph |
| `ex-interactive-vis-network.html` | Standalone vis.js export of an interactive network |

Try notebooks in the browser via [Colab](https://colab.research.google.com/github/kazuyamagiwa/graphnet-automata/blob/master/notebooks/ga01-odd-even-seed-bonding-communities.ipynb) or [Binder](https://mybinder.org/v2/gh/kazuyamagiwa/graphnet-automata/HEAD). For a slider-based UI, run `uv run streamlit run app.py` (see the docs **Interactive demos** page).

Frozen NetworkX 2-era snapshots (original short names) are under [`archive/`](archive/).
