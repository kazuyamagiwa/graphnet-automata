"""Find kernels that produce low degree-distribution entropy.

Entropy is computed on the normalized histogram of *non-zero* degrees.
Low entropy means the degree mass is concentrated on few values, which is a
useful proxy for ordered / regular structure.

Graphs that are mostly edgeless are rejected by assigning a large sentinel
entropy (``100``) so they do not look artificially "ordered".
"""

from __future__ import annotations

import collections
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import entropy

from graphnet_automata.generator import GeneratorState, kernel_from_index


def degree_entropy(graph: nx.Graph) -> tuple[float, list[int], list[int]]:
    """Score a graph's degree histogram and return plotting data.

    Parameters
    ----------
    graph:
        Evolved NetworkX graph.

    Returns
    -------
    ent:
        Shannon entropy of the non-zero degree distribution, or ``100.0`` when
        the graph is filtered out as too sparse / empty.
    deg_init:
        Degree values from the full histogram (including degree 0).
    cnt_init:
        Matching counts for ``deg_init``.
    """
    degree_sequence = sorted((d for _, d in graph.degree()), reverse=True)
    degree_count = collections.Counter(degree_sequence)
    degree_count_init = collections.Counter(degree_sequence)
    deg_init, cnt_init = zip(*degree_count_init.items())

    # Fraction of nodes in the lowest-degree bin of the sorted histogram.
    # Because the sequence is reverse-sorted, the last bin is the smallest
    # degree present (often 0).
    cnt_avg_init_0 = cnt_init[-1] / sum(cnt_init)

    # Work on a copy of the counts that excludes isolated / zero-degree mass.
    del degree_count[0]

    if deg_init[-1] == 0 and cnt_avg_init_0 > 0.10:
        # Too many edge-less nodes: treat as uninteresting for this search.
        ent = 100.0
    elif len(degree_count) > 0:
        _, cnt = zip(*degree_count.items())
        cnt_avg = [cnt[i] / sum(cnt) for i in range(len(cnt))]
        ent = float(entropy(cnt_avg))
    else:
        # No non-zero degrees remain after filtering.
        ent = 100.0

    return ent, list(deg_init), list(cnt_init)


def main(
    nodes: int = 33,
    prob: float = 0.01,
    steps: int = 100,
    entropy_threshold: float = 2.0,
    output_dir: str | Path = ".",
) -> None:
    """Sweep kernels and save histograms for low-entropy hits.

    Parameters
    ----------
    nodes, prob, steps:
        Seed-graph and evolution settings.
    entropy_threshold:
        Save a figure when :func:`degree_entropy` returns a value below this.
    output_dir:
        Destination directory for histogram PNGs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(512):
        print(i)
        kernel = kernel_from_index(i)
        gen = GeneratorState(nodes=nodes, prob=prob, kernel=kernel, steps=steps)
        gen_g1 = nx.from_numpy_array(gen.run())
        ent, deg_init, cnt_init = degree_entropy(gen_g1)

        if ent < entropy_threshold:
            print("hit")
            fig, ax = plt.subplots()
            plt.bar(deg_init, cnt_init, width=0.80, color="b")
            plt.title("Degree Histogram")
            plt.ylabel("Count")
            plt.xlabel("Degree")
            ax.set_xticks([d + 0.4 for d in deg_init])
            ax.set_xticklabels(deg_init)

            # Inset graph drawing for quick visual inspection.
            plt.axes([0.4, 0.4, 0.5, 0.5])
            pos = nx.spring_layout(gen_g1)
            plt.axis("off")
            nx.draw_networkx_nodes(gen_g1, pos, node_size=20)
            nx.draw_networkx_edges(gen_g1, pos, alpha=0.4)
            fig.savefig(output_dir / f"{i}_degree_histogram.png")
            plt.close(fig)


if __name__ == "__main__":
    main()
