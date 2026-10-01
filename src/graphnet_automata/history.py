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


def triptych_indices(steps: int) -> tuple[int, int, int]:
    """Return ``(seed, mid, final)`` generation indices for a three-frame view.

    Mid is ``steps // 2`` (halfway through evolution). For ``steps == 0`` all
    three indices collapse to ``0``.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    return 0, steps // 2, steps


def entropy_band(entropy: float, ordered_threshold: float = 2.0) -> str:
    """Classify degree entropy into a short label for captions.

    Uses the same default cutoff as ``graphnet-automata-search`` (``< 2`` ⇒
    ordered). Sentinel values map to ``sparse``.
    """
    if entropy >= ENTROPY_SENTINEL:
        return "sparse"
    if entropy < ordered_threshold:
        return "ordered"
    if entropy < ordered_threshold + 0.5:
        return "mixed"
    return "disordered"


def _modal_nonzero_degree(adjacency: np.ndarray) -> int | None:
    """Most common non-zero degree, or ``None`` if the graph has no edges."""
    graph = graph_from_adjacency(adjacency)
    nonzero = [d for _, d in graph.degree() if d > 0]
    if not nonzero:
        return None
    values, counts = np.unique(nonzero, return_counts=True)
    return int(values[int(np.argmax(counts))])


def evolution_narrative(
    history: dict[str, Any],
    *,
    seed_nodes: int,
) -> str:
    """Build a 1–2 sentence caption from final / mid metrics.

    Mentions odd/even seed parity, community count when available, degree
    entropy band (ordered vs disordered), and optional lock-in step or modal
    degree when the final graph looks ordered.
    """
    steps = int(history["steps"])
    _, mid_t, final_t = triptych_indices(steps)
    parity = "Odd" if seed_nodes % 2 else "Even"
    ent_final = float(history["degree_entropy"][final_t])
    band = entropy_band(ent_final)
    communities = history["community_count"]
    comm_final = communities[final_t] if communities else None
    lock_t = lock_in_step(history["degree_entropy"])

    parts: list[str] = [f"{parity} seed (n={seed_nodes})"]

    if comm_final is not None:
        comm_mid = communities[mid_t]
        if comm_mid is not None and mid_t != final_t and comm_mid != comm_final:
            parts.append(
                f"communities {comm_mid} → {comm_final} from mid to final"
            )
        else:
            parts.append(f"{comm_final} Louvain "
                         f"{'community' if comm_final == 1 else 'communities'}")

    if band == "sparse":
        parts.append("final graph too sparse for a reliable entropy score")
    else:
        parts.append(f"entropy {ent_final:.2f} ({band})")

    sentence1 = "; ".join(parts) + "."

    extras: list[str] = []
    if band == "ordered":
        mode_deg = _modal_nonzero_degree(history["adjacencies"][final_t])
        if mode_deg is not None:
            extras.append(f"Edge mass concentrated near degree d≈{mode_deg}")
    if lock_t is not None and band != "sparse":
        extras.append(f"structure lock-in around t={lock_t}")
    avg_cnt = float(history["avg_hist_count"][final_t])
    if band == "ordered" and avg_cnt >= 4.0:
        extras.append(f"mean hist count {avg_cnt:.1f}")

    if not extras:
        return sentence1
    # Keep the caption to roughly two sentences.
    sentence2 = extras[0] + ("." if len(extras) == 1 else f"; {extras[1]}.")
    return f"{sentence1} {sentence2}"
