# Interactive demos

Try graphnet-automata without installing everything locally.

## Google Colab

Open the starter notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kazuyamagiwa/graphnet-automata/blob/master/notebooks/ga01-odd-even-seed-bonding-communities.ipynb)

Other notebooks under `notebooks/` can be opened the same way by swapping the path in the Colab URL.

## Binder

Launch a temporary JupyterLab environment from this repository:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kazuyamagiwa/graphnet-automata/HEAD)

Binder installs dependencies from `requirements.txt` and uses Python 3.12
(`runtime.txt`). Cold starts can take a few minutes.

## Streamlit web app

`app.py` is a small UI over {class}`~graphnet_automata.generator.GeneratorState`:

* choose seed size, edge probability, kernel index, and steps
* evolve **one** kernel (kept modest for interactivity)
* view the graph, optional Louvain communities, and degree histogram

Primary controls sit on the main page (with presets) so phones do not need the
sidebar. Results use tabs so graph and histogram stay readable on narrow
screens; desktop keeps the same focused layout.

### Run locally

```bash
uv sync --extra app
uv run streamlit run app.py
```

### Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Visit https://streamlit.io/cloud and sign in with GitHub.
3. **Create app** → select the repo → main file `app.py`.
4. Deploy. Community Cloud installs from root `requirements.txt`.

Keep ``steps`` modest in the UI. Full 512-kernel sweeps and Optuna studies
are better run via the CLI tools documented in {doc}`usage`.
