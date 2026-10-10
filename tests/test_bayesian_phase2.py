"""Phase-2 synthetic DAG recovery smoke / documentation test.

Prints ground-truth vs final adjacency when run with ``pytest -s``.
Asserts the CA loop is stable (shapes, downhill, binary) and that noise
edges shrink vs the random upper-tri start on the default seed.
"""

from __future__ import annotations

import numpy as np

from graphnet_automata.bayesian_synth import (
    NODE_NAMES,
    TRUTH_EDGES,
    format_edges,
    report,
    run_phase2,
    truth_adjacency,
)


def test_truth_dag_has_chain_and_v_structure() -> None:
    A = truth_adjacency()
    # Chain A→B→C
    assert A[0, 1] == 1.0 and A[1, 2] == 1.0
    # V-structure D→F←E
    assert A[3, 5] == 1.0 and A[4, 5] == 1.0
    assert len(TRUTH_EDGES) == 8
    assert len(NODE_NAMES) == 10


def test_phase2_step_loop_shapes_and_report(capsys) -> None:
    result = run_phase2(ticks=60, n_samples=3000, seed=0)
    print(report(result))
    captured = capsys.readouterr()
    assert "Ground-truth edges" in captured.out
    assert "Final directed" in captured.out

    assert result.A_final.shape == (10, 10)
    assert result.E_final.shape == (10,)
    assert set(np.unique(result.A_final)).issubset({0.0, 1.0})
    assert np.all(np.diag(result.A_final) == 0.0)
    assert np.all((result.E_final >= 0.0) & (result.E_final <= 1.0))

    src, dst = np.nonzero(result.A_final)
    if src.size:
        assert np.all(result.E_final[src] > result.E_final[dst])

    # Default seed: noise edges should not grow vs init (pruning signal).
    assert result.metrics_final.fp <= result.metrics_init.fp
    assert format_edges(result.A_true).count("→") == len(TRUTH_EDGES)
