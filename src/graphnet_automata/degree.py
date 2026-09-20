"""Search kernels whose evolved graphs have a high average degree-count."""

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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(512):
        print(i)
        kernel = kernel_from_index(i)
        gen = GeneratorState(nodes=nodes, prob=prob, kernel=kernel, steps=steps)
        gen_g1 = nx.from_numpy_array(gen.run())

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
            plt.axes([0.4, 0.4, 0.5, 0.5])
            pos = nx.spring_layout(gen_g1)
            plt.axis("off")
            nx.draw_networkx_nodes(gen_g1, pos, node_size=20)
            nx.draw_networkx_edges(gen_g1, pos, alpha=0.4)
            fig.savefig(output_dir / f"{i}_degree_histogram.png")
            plt.close(fig)


if __name__ == "__main__":
    main()
