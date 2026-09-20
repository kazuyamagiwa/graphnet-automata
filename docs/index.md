# graphnet-automata

Study of graph evolution using cellular automaton-like methods.

```{toctree}
:maxdepth: 2
:caption: Contents

concept
installation
usage
api
```

## Overview

**graphnet-automata** starts from a small seed graph, converts it to an
adjacency matrix, and evolves that matrix with padded convolution rules
inspired by cellular automata. Different 3×3 binary kernels produce very
different large-scale graphs—bonded pairs, whip-like structures, and other
patterns explored in the project notebooks.

## Quick links

* {doc}`concept` — how the evolution rule works
* {doc}`installation` — install with uv
* {doc}`usage` — CLI tools and Python API
* {doc}`api` — autodoc reference

## Project resources

* Source: https://github.com/kazuyamagiwa/graphnet-automata
* License: MIT
