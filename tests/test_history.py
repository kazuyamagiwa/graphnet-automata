"""Tests for step-wise evolution history used by the Streamlit scrubber."""

from graphnet_automata.history import (
    ENTROPY_SENTINEL,
    collect_evolution_history,
    entropy_band,
    evolution_narrative,
    lock_in_step,
    positions_for_step,
    triptych_indices,
)


def test_collect_evolution_history_length_and_growth() -> None:
    steps = 4
    nodes = 5
    history = collect_evolution_history(
        nodes=nodes,
        prob=0.2,
        kernel_index=21,
        steps=steps,
        graph_seed=0,
        directed=True,
        compute_communities=False,
    )
    assert len(history["adjacencies"]) == steps + 1
    assert history["n_nodes"][0] == nodes
    assert history["n_nodes"][-1] == nodes + 2 * steps
    # |V| grows by exactly 2 each generation.
    assert history["n_nodes"] == [nodes + 2 * t for t in range(steps + 1)]
    assert len(history["degree_entropy"]) == steps + 1
    assert len(history["avg_hist_count"]) == steps + 1
    assert all(c is None for c in history["community_count"])


def test_collect_evolution_history_communities_optional() -> None:
    history = collect_evolution_history(
        nodes=6,
        prob=0.3,
        kernel_index=21,
        steps=3,
        graph_seed=1,
        directed=True,
        compute_communities=True,
    )
    assert all(c is not None for c in history["community_count"])


def test_positions_for_step_offset_mapping() -> None:
    # Fake final positions for a 5-node final graph after 2 steps from n=1.
    final_pos = {i: (float(i), 0.0) for i in range(5)}
    pos_t0 = positions_for_step(final_pos, n_at_t=1, total_steps=2, t=0)
    assert pos_t0 == {0: (2.0, 0.0)}
    pos_t1 = positions_for_step(final_pos, n_at_t=3, total_steps=2, t=1)
    assert pos_t1 == {0: (1.0, 0.0), 1: (2.0, 0.0), 2: (3.0, 0.0)}


def test_lock_in_step_detects_drop() -> None:
    series = [3.0, 2.9, 2.8, 1.0, 0.9, 0.85]
    assert lock_in_step(series) == 3
    assert lock_in_step([2.0, 2.0, 2.0]) is None
    assert lock_in_step([100.0, 100.0, 1.0, 0.9]) == 2


def test_triptych_indices() -> None:
    assert triptych_indices(0) == (0, 0, 0)
    assert triptych_indices(1) == (0, 0, 1)
    assert triptych_indices(10) == (0, 5, 10)
    assert triptych_indices(11) == (0, 5, 11)


def test_entropy_band() -> None:
    assert entropy_band(ENTROPY_SENTINEL) == "sparse"
    assert entropy_band(1.4) == "ordered"
    assert entropy_band(2.2) == "mixed"
    assert entropy_band(3.0) == "disordered"


def test_evolution_narrative_odd_ordered() -> None:
    history = collect_evolution_history(
        nodes=13,
        prob=0.05,
        kernel_index=448,
        steps=20,
        graph_seed=1,
        directed=True,
        compute_communities=True,
    )
    text = evolution_narrative(history, seed_nodes=13)
    assert text.startswith("Odd seed (n=13)")
    assert "entropy" in text
    assert "(ordered)" in text
    # Caption should stay short (about 1–2 sentences).
    assert len(text) < 280
    assert "Louvain" in text or "communities" in text
    assert text.endswith(".")

def test_evolution_narrative_even_parity() -> None:
    history = collect_evolution_history(
        nodes=8,
        prob=0.15,
        kernel_index=21,
        steps=6,
        graph_seed=1,
        directed=True,
        compute_communities=False,
    )
    text = evolution_narrative(history, seed_nodes=8)
    assert text.startswith("Even seed (n=8)")
    # Without community scoring, caption still reports entropy band or sparse.
    assert "entropy" in text or "sparse" in text
