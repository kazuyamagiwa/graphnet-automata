"""graphnet-automata: evolve graphs with cellular automaton-like rules."""

from graphnet_automata.generator import GeneratorState, kernel_from_index

__all__ = ["GeneratorState", "kernel_from_index", "main"]


def main() -> None:
    """Default CLI entry point; lists available commands."""
    print(
        "graphnet-automata commands:\n"
        "  graphnet-automata-degree    Search kernels by degree-count average\n"
        "  graphnet-automata-generate  Generate an HDF5 dataset of evolved graphs\n"
        "  graphnet-automata-optimize  Optimize seed parameters with Optuna\n"
        "  graphnet-automata-search    Search kernels by degree-distribution entropy"
    )
