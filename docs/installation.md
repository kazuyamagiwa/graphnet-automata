# Installation

## Requirements

* Python 3.10+
* [uv](https://docs.astral.sh/uv/) (recommended)

## Install with uv

From the repository root:

```bash
uv sync
```

Optional extras:

```bash
# notebooks / tests
uv sync --extra dev

# pyvis interactive graphs (used by some notebooks)
uv sync --extra viz

# build this documentation locally
uv sync --extra docs
```

## Editable check

```bash
uv run python -c "from graphnet_automata import GeneratorState; print(GeneratorState)"
```

## Build the docs locally

```bash
uv sync --extra docs
uv run sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser.

## Read the Docs hosting

This repository includes a `.readthedocs.yaml` config. To publish:

1. Import the GitHub repo at https://readthedocs.org/
2. Use the default branch (or this feature branch while reviewing)
3. Read the Docs will install the `docs` extra and run Sphinx automatically
