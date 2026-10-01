"""Step-wise evolution history for interactive playback and metric strips.

Used by the Streamlit app to scrub generations and plot growth / order scores
without re-implementing the CA update or CLI scoring helpers.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from graphnet_automata.generate import adj2avgcnt
from graphnet_automata.generator import GeneratorState, kernel_from_index
from graphnet_automata.search import degree_entropy

# Sentinel from search.degree_entropy for too-sparse graphs.
ENTROPY_SENTINEL = 100.0


def graph_from_adjacency(adjacency: np.ndarray) -> nx.Graph:
    """Build an undirected graph from an adjacency matrix (viz / scoring path)."""
    graph = nx.Graph()
    graph.add_nodes_from(range(adjacency.shape[0]))
    edges = list(nx.from_numpy_array(adjacency).edges())
    graph.add_edges_from(edges)
    return graph


def community_count(graph: nx.Graph) -> int | None:
    """Return Louvain community count, or ``None`` if detection fails."""
    if graph.number_of_edges() == 0:
        return 0
    try:
        import community as community_louvain

        partition = community_louvain.best_partition(graph)
        return len(set(partition.values()))
    except Exception:
        return None


def collect_evolution_history(
    nodes: int,
    prob: float,
    kernel_index: int,
    steps: int,
    graph_seed: int,
    directed: bool,
    compute_communities: bool = False,
) -> dict[str, Any]:
    """Evolve with ``next_state`` and return per-step snapshots + metrics.

    Index ``0`` is the seed adjacency; indices ``1…steps`` are post-update
    matrices. Final spring-layout positions are included so callers can
    subset them for earlier frames without layout jumpiness.
    """
    kernel_local = kernel_from_index(kernel_index)
    gen = GeneratorState(
        nodes=nodes,
        prob=prob,
        kernel=kernel_local,
        steps=steps,
        graph_seed=graph_seed,
        directed=directed,
    )

    n_nodes_series: list[int] = []
    n_edges_series: list[int] = []
    avg_degree_series: list[float] = []
    entropy_series: list[float] = []
    avg_count_series: list[float] = []
    community_series: list[int | None] = []
    adjacencies: list[np.ndarray] = []

    def _record(adj: np.ndarray) -> None:
        graph = graph_from_adjacency(adj)
        degrees = [d for _, d in graph.degree()]
        n_nodes_series.append(graph.number_of_nodes())
        n_edges_series.append(graph.number_of_edges())
        avg_degree_series.append(float(np.mean(degrees)) if degrees else 0.0)
        ent, _, _ = degree_entropy(graph)
        entropy_series.append(float(ent))
        avg_count_series.append(float(adj2avgcnt(adj)))
        if compute_communities:
            community_series.append(community_count(graph))
        else:
            community_series.append(None)
        adjacencies.append(np.array(adj, copy=True))

    _record(gen.seed)
    for _ in range(steps):
        _record(gen.next_state())

    final_graph = graph_from_adjacency(adjacencies[-1])
    final_pos = nx.spring_layout(final_graph, seed=int(graph_seed))
    pos_serial = {int(n): (float(xy[0]), float(xy[1])) for n, xy in final_pos.items()}

    return {
        "adjacencies": adjacencies,
        "n_nodes": n_nodes_series,
        "n_edges": n_edges_series,
        "avg_degree": avg_degree_series,
        "degree_entropy": entropy_series,
        "avg_hist_count": avg_count_series,
        "community_count": community_series,
        "final_pos": pos_serial,
        "steps": steps,
    }


def positions_for_step(
    final_pos: dict[int, tuple[float, float]],
    n_at_t: int,
    total_steps: int,
    t: int,
) -> dict[int, tuple[float, float]]:
    """Map final-layout coordinates onto the generation-``t`` vertex set.

    Each ``next_state`` pads by one row/column on every side, so vertex ``i``
    at step ``t`` is vertex ``i + (total_steps - t)`` in the final matrix.
    """
    offset = total_steps - t
    return {
        i: final_pos[i + offset]
        for i in range(n_at_t)
        if (i + offset) in final_pos
    }


def lock_in_step(entropy_series: list[float], min_drop: float = 0.25) -> int | None:
    """Return the step after the largest meaningful entropy drop, if any.

    Sparse-graph sentinel values (``ENTROPY_SENTINEL``) are kept as a high
    ceiling so a transition into a real entropy band counts as locking in.
    """
    if len(entropy_series) < 2:
        return None
    arr = np.asarray(entropy_series, dtype=float)
    deltas = np.diff(arr)
    drop_idx = int(np.argmin(deltas))
    if deltas[drop_idx] >= -min_drop:
        return None
    return drop_idx + 1
