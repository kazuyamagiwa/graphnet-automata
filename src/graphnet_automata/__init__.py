"""Public package surface for graphnet-automata.

graphnet-automata evolves small "seed" graphs by treating their adjacency
matrices like cellular-automaton grids. Each generation pads the matrix,
counts neighbors with a 3x3 binary kernel, and applies birth/survival rules.

This module re-exports the core types used by notebooks and scripts, and
provides the default ``graphnet-automata`` CLI entry point.
"""

from graphnet_automata.generator import GeneratorState, kernel_from_index

__all__ = ["GeneratorState", "kernel_from_index", "main"]


def main() -> None:
    """Print the installed CLI commands.

    The package exposes several console scripts via ``pyproject.toml``.
    Running ``graphnet-automata`` with no subcommand lists them so users can
    discover the available workflows after ``uv sync``.
    """
    print(
        "graphnet-automata commands:\n"
        "  graphnet-automata-degree    Search kernels by degree-count average\n"
        "  graphnet-automata-generate  Generate an HDF5 dataset of evolved graphs\n"
        "  graphnet-automata-optimize  Optimize seed parameters with Optuna\n"
        "  graphnet-automata-search    Search kernels by degree-distribution entropy"
    )
