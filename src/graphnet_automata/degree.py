"""Sweep kernels and keep graphs with a high average degree-count.

This tool is the packaged form of the original ``graphnet-automata.py``
script. It tries every 3x3 binary kernel (``0..511``), evolves a seed graph,
builds the degree histogram, and saves a figure when the average bin count
crosses a threshold.

A high average count tends to mean the histogram is concentrated on few
degree values (many nodes share similar degrees), which is one simple way to
spot "ordered" or structured outcomes among the 512 kernels.
"""

from __future__ import annotations

import collections
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from graphnet_automata.generator import GeneratorState, kernel_from_index


def main(
    nodes: int = 13,
    prob: float = 0.05,
    steps: int = 300,
    avg_count_threshold: float = 10.0,
    output_dir: str | Path = ".",
) -> None:
    """Search kernels by average degree-histogram count and save hits.

    Parameters
    ----------
    nodes:
        Size of the Erdős–Rényi seed graph.
    prob:
        Edge probability for the seed graph.
    steps:
        Automaton generations before scoring.
    avg_count_threshold:
        Save a plot when ``mean(histogram_counts)`` exceeds this value.
    output_dir:
        Directory for ``{kernel}_degree_histogram.png`` files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Exhaustive sweep of the 9-bit kernel space.
    for i in range(512):
        print(i)
        kernel = kernel_from_index(i)
        gen = GeneratorState(nodes=nodes, prob=prob, kernel=kernel, steps=steps)
        gen_g1 = nx.from_numpy_array(gen.run())

        # Degree histogram: degrees on the x-axis, node counts on the y-axis.
        degree_sequence = sorted((d for _, d in gen_g1.degree()), reverse=True)
        degree_count = collections.Counter(degree_sequence)
        deg, cnt = zip(*degree_count.items())

        if np.average(cnt) > avg_count_threshold:
            print("hit")
            fig, ax = plt.subplots()
            plt.bar(deg, cnt, width=0.80, color="b")
            plt.title("Degree Histogram")
            plt.ylabel("Count")
            plt.xlabel("Degree")
            ax.set_xticks([d + 0.4 for d in deg])
            ax.set_xticklabels(deg)

            # Inset spring-layout drawing of the evolved graph.
            plt.axes([0.4, 0.4, 0.5, 0.5])
            pos = nx.spring_layout(gen_g1)
            plt.axis("off")
            nx.draw_networkx_nodes(gen_g1, pos, node_size=20)
            nx.draw_networkx_edges(gen_g1, pos, alpha=0.4)
            fig.savefig(output_dir / f"{i}_degree_histogram.png")
            plt.close(fig)


if __name__ == "__main__":
    main()
