"""Core graphnet automaton generator."""

from __future__ import annotations

import numpy as np
import networkx as nx
import scipy.ndimage as nd


def kernel_from_index(index: int) -> np.ndarray:
    """Build a 3x3 binary convolution kernel from an integer in ``[0, 511]``."""
    kernel_seed = f"{index:09b}"
    return np.array(list(kernel_seed), dtype=np.uint8).reshape((3, 3))


class GeneratorState:
    """Evolve an adjacency matrix by padded convolution with a binary kernel."""

    def __init__(
        self,
        nodes: int = 13,
        prob: float = 0.05,
        kernel: np.ndarray | None = None,
        steps: int = 100,
        graph_seed: int = 1,
        directed: bool = True,
    ) -> None:
        self.nodes = nodes
        self.prob = prob
        self.kernel = (
            np.zeros((3, 3), dtype=np.uint8) if kernel is None else np.asarray(kernel)
        )
        self.steps = steps
        self.seed = nx.to_numpy_array(
            nx.erdos_renyi_graph(
                self.nodes, self.prob, seed=graph_seed, directed=directed
            )
        )

    def next_state(self) -> np.ndarray:
        seed = np.pad(self.seed, (1, 1), "constant")
        neighbor_count = nd.convolve(seed, self.kernel, mode="constant")
        self.seed = np.where(
            ((seed == 0) & (neighbor_count > 0) & (neighbor_count <= 4))
            | ((seed == 1) & (neighbor_count > 4)),
            1,
            0,
        )
        return self.seed

    def run(self, steps: int | None = None) -> np.ndarray:
        n_steps = self.steps if steps is None else steps
        for _ in range(n_steps):
            self.next_state()
        return self.seed
