"""Phase-2 synthetic linear-DAG experiment for :func:`bayesian.step`.

Generates N=10 continuous variables from a known linear SEM (chain +
V-structure + extras), estimates empirical covariance, initializes a random
upper-triangular adjacency and random elevations, then loops the NumPy CA
``step`` and reports recovery vs ground truth.

Run::

    uv run python -m graphnet_automata.bayesian_synth
    uv run python -m graphnet_automata.bayesian_synth --seeds 10
    uv run pytest tests/test_bayesian_phase2.py -s
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from graphnet_automata.bayesian import step

# Node labels for readable reporting (index = variable id).
NODE_NAMES = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")
N = 10

# Ground-truth directed edges i→j (parent → child). All edges satisfy i < j
# so index order is a valid topological order.
#
#   Chain A→B→C:        0→1→2
#   V-structure D→F←E:  3→5←4
#   Chain G→H→I:        6→7→8
#   Collider into J:    2→9, 8→9
TRUTH_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (3, 5),
    (4, 5),
    (6, 7),
    (7, 8),
    (2, 9),
    (8, 9),
)


def truth_adjacency() -> np.ndarray:
    """Binary ground-truth adjacency ``(N, N)``."""
    A = np.zeros((N, N), dtype=np.float64)
    for i, j in TRUTH_EDGES:
        A[i, j] = 1.0
    return A


def generate_linear_dag_samples(
    n_samples: int = 5000,
    noise_scale: float = 1.0,
    weight: float = 2.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw samples from the linear SEM implied by :data:`TRUTH_EDGES`.

    Each node is ``X_j = sum_i W[i,j] X_i + eps_j`` with fixed weight ``weight``
    on every true edge and i.i.d. Gaussian noise.

    Returns
    -------
    X, W
        ``X`` has shape ``(n_samples, N)``; ``W`` is the SEM weight matrix.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    W = np.zeros((N, N), dtype=np.float64)
    for i, j in TRUTH_EDGES:
        W[i, j] = weight

    # Forward sweep in index order (valid topo order for this DAG).
    X = np.zeros((n_samples, N), dtype=np.float64)
    for j in range(N):
        signal = X @ W[:, j]
        X[:, j] = signal + rng.normal(0.0, noise_scale, size=n_samples)
    return X, W


def empirical_covariance(X: np.ndarray) -> np.ndarray:
    """Unbiased sample covariance ``(N, N)`` with row observations."""
    Xc = X - X.mean(axis=0, keepdims=True)
    n = X.shape[0]
    return (Xc.T @ Xc) / max(n - 1, 1)


def random_upper_tri_adjacency(
    n: int,
    p: float = 0.4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Random binary upper-triangular adjacency (strict, zero diagonal)."""
    if rng is None:
        rng = np.random.default_rng(0)
    A = (rng.random((n, n)) < p).astype(np.float64)
    return np.triu(A, k=1)


def undirected_skeleton(A: np.ndarray) -> np.ndarray:
    """Symmetric 0/1 adjacency with zero diagonal."""
    S = ((A + A.T) > 0).astype(np.float64)
    np.fill_diagonal(S, 0.0)
    return S


@dataclass(frozen=True)
class EdgeMetrics:
    """Edge recovery counts vs ground truth."""

    tp: int
    fp: int
    fn: int
    true_edges: int
    pred_edges: int

    @property
    def precision(self) -> float:
        return self.tp / self.pred_edges if self.pred_edges else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.true_edges if self.true_edges else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def edge_metrics(A_pred: np.ndarray, A_true: np.ndarray) -> EdgeMetrics:
    """TP / FP / FN on directed edges (diagonal ignored)."""
    pred = (A_pred > 0) & ~np.eye(A_pred.shape[0], dtype=bool)
    true = (A_true > 0) & ~np.eye(A_true.shape[0], dtype=bool)
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    return EdgeMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        true_edges=int(np.sum(true)),
        pred_edges=int(np.sum(pred)),
    )


def skeleton_metrics(A_pred: np.ndarray, A_true: np.ndarray) -> EdgeMetrics:
    """Undirected skeleton TP/FP/FN (each unordered pair counted once)."""
    pred = np.triu(undirected_skeleton(A_pred) > 0, k=1)
    true = np.triu(undirected_skeleton(A_true) > 0, k=1)
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    return EdgeMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        true_edges=int(np.sum(true)),
        pred_edges=int(np.sum(pred)),
    )


def format_edges(A: np.ndarray) -> str:
    """Human-readable edge list using :data:`NODE_NAMES`."""
    src, dst = np.nonzero(A > 0)
    if src.size == 0:
        return "(none)"
    parts = [f"{NODE_NAMES[i]}→{NODE_NAMES[j]}" for i, j in zip(src, dst)]
    return ", ".join(parts)


@dataclass
class Phase2Result:
    """Artifacts from one Phase-2 run."""

    A_true: np.ndarray
    A_init: np.ndarray
    A_final: np.ndarray
    E_init: np.ndarray
    E_final: np.ndarray
    Cov: np.ndarray
    metrics_init: EdgeMetrics
    metrics_final: EdgeMetrics
    skeleton_init: EdgeMetrics
    skeleton_final: EdgeMetrics
    ticks: int
    history: list[EdgeMetrics]
    seed: int


def run_phase2(
    *,
    n_samples: int = 5000,
    ticks: int = 100,
    edge_prob: float = 0.4,
    weight: float = 2.0,
    learning_rate: float = 0.1,
    threshold_death: float = 0.05,
    threshold_birth: float = 0.15,
    k: int = 2,
    seed: int = 0,
) -> Phase2Result:
    """Generate data, init state, loop ``step``, return comparison artifacts."""
    rng = np.random.default_rng(seed)
    X, _ = generate_linear_dag_samples(
        n_samples=n_samples, weight=weight, rng=rng
    )
    Cov = empirical_covariance(X)
    A_true = truth_adjacency()
    A = random_upper_tri_adjacency(N, p=edge_prob, rng=rng)
    E = rng.random(N)

    metrics_init = edge_metrics(A, A_true)
    sk_init = skeleton_metrics(A, A_true)
    history = [metrics_init]
    A_init = A.copy()
    E_init = E.copy()

    for _ in range(ticks):
        A, E = step(
            A,
            E,
            Cov,
            learning_rate=learning_rate,
            threshold_death=threshold_death,
            threshold_birth=threshold_birth,
            k=k,
        )
        history.append(edge_metrics(A, A_true))

    return Phase2Result(
        A_true=A_true,
        A_init=A_init,
        A_final=A,
        E_init=E_init,
        E_final=E,
        Cov=Cov,
        metrics_init=metrics_init,
        metrics_final=history[-1],
        skeleton_init=sk_init,
        skeleton_final=skeleton_metrics(A, A_true),
        ticks=ticks,
        history=history,
        seed=seed,
    )


def report(result: Phase2Result) -> str:
    """Multi-line human-readable summary for stdout / pytest ``-s``."""
    m0, m1 = result.metrics_init, result.metrics_final
    s0, s1 = result.skeleton_init, result.skeleton_final
    lines = [
        "=== Phase 2 synthetic Bayesian CA ===",
        f"seed={result.seed}  N={N}  ticks={result.ticks}",
        f"Ground-truth edges ({m1.true_edges}): {format_edges(result.A_true)}",
        f"Init A edges ({m0.pred_edges}): {format_edges(result.A_init)}",
        f"Init directed:  TP={m0.tp} FP={m0.fp} FN={m0.fn}  "
        f"P={m0.precision:.2f} R={m0.recall:.2f} F1={m0.f1:.2f}",
        f"Init skeleton:  TP={s0.tp} FP={s0.fp} FN={s0.fn}  "
        f"P={s0.precision:.2f} R={s0.recall:.2f} F1={s0.f1:.2f}",
        f"Final A edges ({m1.pred_edges}): {format_edges(result.A_final)}",
        f"Final directed: TP={m1.tp} FP={m1.fp} FN={m1.fn}  "
        f"P={m1.precision:.2f} R={m1.recall:.2f} F1={m1.f1:.2f}",
        f"Final skeleton: TP={s1.tp} FP={s1.fp} FN={s1.fn}  "
        f"P={s1.precision:.2f} R={s1.recall:.2f} F1={s1.f1:.2f}",
        f"Noise prune ΔFP (directed): {m0.fp} → {m1.fp} ({m1.fp - m0.fp:+d})",
        f"E_init:  {np.array2string(result.E_init, precision=3)}",
        f"E_final: {np.array2string(result.E_final, precision=3)}",
        "True edge status (final):",
    ]
    for i, j in TRUTH_EDGES:
        present = result.A_final[i, j] > 0
        mark = "HIT" if present else "MISS"
        lines.append(f"  {NODE_NAMES[i]}→{NODE_NAMES[j]}: {mark}")
    return "\n".join(lines)


def report_multi(results: list[Phase2Result]) -> str:
    """Aggregate directed / skeleton metrics across seeds."""
    dir_f1 = np.array([r.metrics_final.f1 for r in results])
    dir_tp = np.array([r.metrics_final.tp for r in results], dtype=float)
    dir_fp = np.array([r.metrics_final.fp for r in results], dtype=float)
    dir_fn = np.array([r.metrics_final.fn for r in results], dtype=float)
    sk_f1 = np.array([r.skeleton_final.f1 for r in results])
    init_fp = np.array([r.metrics_init.fp for r in results], dtype=float)
    lines = [
        f"=== Multi-seed summary (n={len(results)}) ===",
        f"Directed final:  mean TP/FP/FN = "
        f"{dir_tp.mean():.2f}/{dir_fp.mean():.2f}/{dir_fn.mean():.2f}  "
        f"mean F1={dir_f1.mean():.2f} ± {dir_f1.std():.2f}",
        f"Skeleton final:  mean F1={sk_f1.mean():.2f} ± {sk_f1.std():.2f}",
        f"Mean init FP → final FP: {init_fp.mean():.1f} → {dir_fp.mean():.1f}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--ticks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=int,
        default=0,
        help="If >0, run this many seeds (0..seeds-1) and print aggregate.",
    )
    parser.add_argument("--edge-prob", type=float, default=0.4)
    parser.add_argument("--weight", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--threshold-death", type=float, default=0.05)
    parser.add_argument("--threshold-birth", type=float, default=0.15)
    parser.add_argument("--k", type=int, default=2)
    args = parser.parse_args(argv)

    kwargs = dict(
        n_samples=args.samples,
        ticks=args.ticks,
        edge_prob=args.edge_prob,
        weight=args.weight,
        learning_rate=args.learning_rate,
        threshold_death=args.threshold_death,
        threshold_birth=args.threshold_birth,
        k=args.k,
    )

    if args.seeds and args.seeds > 0:
        results = [run_phase2(seed=s, **kwargs) for s in range(args.seeds)]
        print(report(results[0]))
        print()
        print(report_multi(results))
    else:
        print(report(run_phase2(seed=args.seed, **kwargs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
