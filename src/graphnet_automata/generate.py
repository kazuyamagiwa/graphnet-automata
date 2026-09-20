"""Generate an HDF5 dataset of automata-grown graphs."""

from __future__ import annotations

import collections
from pathlib import Path
from statistics import mean

import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from graphnet_automata.generator import GeneratorState, kernel_from_index


def adj2avgcnt(adj: np.ndarray) -> float:
    """Calculate degree-count average from an adjacency matrix."""
    gen = nx.from_numpy_array(adj)
    degree_sequence = sorted((d for _, d in gen.degree()), reverse=True)
    degree_count = collections.Counter(degree_sequence)
    _, cnt = zip(*degree_count.items())
    return mean(cnt)


def main(
    nodes: int = 13,
    prob: float = 0.05,
    steps: int = 100,
    output_file: str | Path = "data/GA_seed_13_0.05_50.h5",
    preview_seed: int = 448,
    show_preview: bool = True,
) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, "w") as h5file:
        for i in range(512):
            print(i)
            kernel = kernel_from_index(i)
            gen = GeneratorState(nodes=nodes, prob=prob, kernel=kernel, steps=steps)
            adj_g1 = gen.run()
            cnt_avg_g1 = adj2avgcnt(adj_g1)
            print(cnt_avg_g1)

            group = f"seed_{i}"
            h5file.create_group(group)
            h5file.create_dataset(
                f"{group}/adjacency_matrix",
                data=adj_g1,
                compression="gzip",
                compression_opts=9,
            )
            h5file.create_dataset(f"{group}/average_count", data=cnt_avg_g1)
            h5file.flush()

    if not show_preview:
        return

    with h5py.File(output_file, "r") as h5file:
        folder = f"seed_{preview_seed}"
        adj1 = h5file[f"{folder}/adjacency_matrix"][()]
        g1 = nx.from_numpy_array(adj1)
        nx.draw(g1, node_size=10, alpha=0.5)
        plt.show()


if __name__ == "__main__":
    main()
