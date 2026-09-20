"""Core graph-evolution engine.

The generator starts from an Erdős–Rényi seed graph, converts it to an
adjacency matrix, then repeatedly:

1. Pads the matrix by one cell on every side (two new nodes appear).
2. Convolves with a 3x3 binary kernel to count local structure.
3. Updates each cell with CA-like birth / survival thresholds.

There are exactly ``2**9 = 512`` possible binary 3x3 kernels. Experiments in
this project usually sweep that full range and score the resulting graphs.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import scipy.ndimage as nd


def kernel_from_index(index: int) -> np.ndarray:
    """Convert an integer in ``[0, 511]`` into a 3x3 binary convolution kernel.

    The integer is written as a 9-bit binary string and reshaped row-major into
    a ``(3, 3)`` array. Bit ``0`` is the top-left kernel weight; bit ``8`` is
    the bottom-right weight.

    Parameters
    ----------
    index:
        Kernel identifier. Values outside ``0..511`` still produce 9 bits via
        formatting, but the intended search space is ``range(512)``.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of shape ``(3, 3)`` with entries in ``{0, 1}``.
    """
    kernel_seed = f"{index:09b}"
    return np.array(list(kernel_seed), dtype=np.uint8).reshape((3, 3))


class GeneratorState:
    """Evolve an adjacency matrix with padded convolution updates.

    Each call to :meth:`next_state` grows the matrix by two rows and two
    columns. After many steps the adjacency matrix becomes a much larger
    graph whose community structure, degree histogram, and entropy can be
    inspected by the search / generate / optimize tools.

    Attributes
    ----------
    nodes:
        Number of nodes in the initial Erdős–Rényi seed graph.
    prob:
        Edge probability for that seed graph.
    kernel:
        3x3 binary convolution kernel used for neighbor counting.
    steps:
        Default number of evolution steps used by :meth:`run`.
    seed:
        Current adjacency matrix (``numpy.ndarray``). Updated in place.
    """

    def __init__(
        self,
        nodes: int = 13,
        prob: float = 0.05,
        kernel: np.ndarray | None = None,
        steps: int = 100,
        graph_seed: int = 1,
        directed: bool = True,
    ) -> None:
        """Create a generator from an Erdős–Rényi adjacency matrix.

        Parameters
        ----------
        nodes:
            Seed graph size.
        prob:
            Seed graph edge probability.
        kernel:
            Optional 3x3 kernel. If omitted, an all-zero kernel is used
            (usually replaced before calling :meth:`run`).
        steps:
            Default evolution length for :meth:`run`.
        graph_seed:
            RNG seed passed to NetworkX so runs are reproducible.
        directed:
            Whether the seed graph is directed. The adjacency matrix is still
            treated as a numeric grid during convolution.
        """
        self.nodes = nodes
        self.prob = prob
        self.kernel = (
            np.zeros((3, 3), dtype=np.uint8) if kernel is None else np.asarray(kernel)
        )
        self.steps = steps
        # NetworkX 3+: to_numpy_array replaces the removed to_numpy_matrix API.
        self.seed = nx.to_numpy_array(
            nx.erdos_renyi_graph(
                self.nodes, self.prob, seed=graph_seed, directed=directed
            )
        )

    def next_state(self) -> np.ndarray:
        """Advance the automaton by one generation.

        Update rule (applied after padding and convolution):

        * empty cell (``0``) becomes ``1`` when neighbor count is in ``1..4``
          ("birth")
        * occupied cell (``1``) stays ``1`` only when neighbor count is ``> 4``
          ("survival"); otherwise it becomes ``0``

        Returns
        -------
        numpy.ndarray
            The updated adjacency matrix (also stored on ``self.seed``).
        """
        # Grow the grid so new nodes can appear around the existing graph.
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
        """Evolve for several generations and return the final matrix.

        Parameters
        ----------
        steps:
            Number of :meth:`next_state` calls. Defaults to ``self.steps``.

        Returns
        -------
        numpy.ndarray
            Final adjacency matrix after evolution.
        """
        n_steps = self.steps if steps is None else steps
        for _ in range(n_steps):
            self.next_state()
        return self.seed
