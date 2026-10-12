"""Synthetic linear-DAG experiments for :func:`bayesian.step`.

Phase 2 (N=10) and Phase 5 scale (N=50) share the same harness: build a known
linear-Gaussian SEM, estimate empirical covariance, initialize a random
upper-triangular adjacency + random elevations, loop NumPy ``step``, and
report directed / skeleton recovery. The step loop is timed with
``time.perf_counter``.

Run::

    uv run python -m graphnet_automata.bayesian_synth
    uv run python -m graphnet_automata.bayesian_synth --seeds 10
    uv run python -m graphnet_automata.bayesian_synth --n 50
    uv run python -m graphnet_automata.bayesian_synth --n 50 --backend jax
    uv run pytest tests/test_bayesian_phase2.py -s
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from graphnet_automata.bayesian import step as numpy_step

# ---------------------------------------------------------------------------
# Phase-2 N=10 ground truth (fixed motifs; kept for tests / default CLI)
# ---------------------------------------------------------------------------

NODE_NAMES = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")
N = 10  # default / Phase-2 size

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


def make_truth_edges(
    n: int,
    *,
    target_edges: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[tuple[int, int], ...]:
    """Build a sparse known DAG with edges ``i→j`` only for ``i < j``.

    * ``n == 10`` — Phase-2 motif set (:data:`TRUTH_EDGES`).
    * larger ``n`` — backbone chain, periodic V-structures / colliders, then
      random forward edges until ``target_edges`` (default ~1.25·n, clipped
      to ``[50, 75]`` for ``50 <= n < 100`` and ``[100, 150]`` for ``n >= 100``).
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if n == 10 and target_edges is None:
        return TRUTH_EDGES

    if rng is None:
        rng = np.random.default_rng(0)
    if target_edges is None:
        if n >= 100:
            target_edges = int(np.clip(round(1.25 * n), 100, 150))
        elif n >= 50:
            target_edges = int(np.clip(round(1.25 * n), 50, 75))
        else:
            target_edges = max(n - 1, min(n + n // 2, n * (n - 1) // 2))

    max_edges = n * (n - 1) // 2
    target_edges = int(min(max(target_edges, n - 1), max_edges))

    edges: set[tuple[int, int]] = set()
    # Backbone chain so every node is reachable in the topo order.
    for i in range(n - 1):
        edges.add((i, i + 1))
    # Periodic V-structures / colliders: i → i+2 ← i+1
    for i in range(0, n - 2, 4):
        edges.add((i, i + 2))
        edges.add((i + 1, i + 2))
    # Occasional longer skip edges for sparsity variety.
    for i in range(0, n - 5, 7):
        edges.add((i, i + 5))

    # Fill with random forward edges to hit the target count.
    while len(edges) < target_edges:
        i = int(rng.integers(0, n - 1))
        j = int(rng.integers(i + 1, n))
        edges.add((i, j))

    # If motifs overshot, drop random non-backbone extras (keep the chain).
    backbone = {(i, i + 1) for i in range(n - 1)}
    extras = sorted(edges - backbone)
    while len(edges) > target_edges and extras:
        drop = extras.pop(int(rng.integers(0, len(extras))))
        edges.discard(drop)
        extras = sorted(edges - backbone)

    return tuple(sorted(edges))


def truth_adjacency(
    n: int = N,
    *,
    edges: tuple[tuple[int, int], ...] | None = None,
) -> np.ndarray:
    """Binary ground-truth adjacency ``(n, n)``."""
    if edges is None:
        edges = make_truth_edges(n)
    A = np.zeros((n, n), dtype=np.float64)
    for i, j in edges:
        A[i, j] = 1.0
    return A


def generate_linear_dag_samples(
    n_samples: int = 5000,
    noise_scale: float = 1.0,
    weight: float = 2.0,
    rng: np.random.Generator | None = None,
    *,
    n: int | None = None,
    edges: tuple[tuple[int, int], ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw samples from a linear SEM on a known DAG.

    Each node is ``X_j = sum_i W[i,j] X_i + eps_j`` with fixed weight ``weight``
    on every true edge and i.i.d. Gaussian noise.

    Returns
    -------
    X, W
        ``X`` has shape ``(n_samples, n)``; ``W`` is the SEM weight matrix.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if edges is None:
        edges = make_truth_edges(n if n is not None else N, rng=rng)
    n_nodes = n if n is not None else (max(max(e) for e in edges) + 1 if edges else N)

    W = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i, j in edges:
        W[i, j] = weight

    # Forward sweep in index order (valid topo order when all edges have i < j).
    X = np.zeros((n_samples, n_nodes), dtype=np.float64)
    for j in range(n_nodes):
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


def _edge_label(i: int, j: int, n: int) -> str:
    if n == 10 and i < len(NODE_NAMES) and j < len(NODE_NAMES):
        return f"{NODE_NAMES[i]}→{NODE_NAMES[j]}"
    return f"{i}→{j}"


def format_edges(A: np.ndarray, *, max_edges: int | None = None) -> str:
    """Human-readable edge list (letter labels for N=10, indices otherwise)."""
    src, dst = np.nonzero(A > 0)
    if src.size == 0:
        return "(none)"
    n = A.shape[0]
    parts = [_edge_label(int(i), int(j), n) for i, j in zip(src, dst)]
    if max_edges is not None and len(parts) > max_edges:
        shown = ", ".join(parts[:max_edges])
        return f"{shown}, … (+{len(parts) - max_edges} more)"
    return ", ".join(parts)


@dataclass
class Phase2Result:
    """Artifacts from one synthetic run (Phase 2 / Phase 5)."""

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
    n: int = N
    truth_edges: tuple[tuple[int, int], ...] = TRUTH_EDGES
    elapsed_sec: float = 0.0
    backend: str = "numpy"
    device: str = "numpy"

    @property
    def ticks_per_sec(self) -> float:
        return self.ticks / self.elapsed_sec if self.elapsed_sec > 0 else float("inf")


def _resolve_step(backend: str) -> tuple[Callable, str]:
    """Return ``(step_fn, device_label)`` for ``numpy`` or ``jax``."""
    if backend == "numpy":
        return numpy_step, "numpy"
    if backend == "jax":
        from graphnet_automata.jax_core import device_info, step as jax_step

        def _step(A, E, Cov, **kwargs):
            import jax.numpy as jnp

            A1, E1 = jax_step(
                jnp.asarray(A),
                jnp.asarray(E),
                jnp.asarray(Cov),
                learning_rate=kwargs.get("learning_rate", 0.1),
                threshold_death=kwargs.get("threshold_death", 0.05),
                threshold_birth=kwargs.get("threshold_birth", 0.15),
                k=kwargs.get("k", 2),
            )
            return np.asarray(A1), np.asarray(E1)

        return _step, device_info()
    raise ValueError(f"Unknown backend {backend!r}; use 'numpy' or 'jax'")


def _default_edge_prob(n: int) -> float:
    """Sparse-ish upper-tri init density (keeps expected edges O(n))."""
    if n <= 10:
        return 0.4
    # Aim for ~2–3× true-edge count as noisy start without filling the tri.
    target_init = min(3.0 * n, 0.15 * n * (n - 1) / 2)
    return float(target_init / (n * (n - 1) / 2))


def _default_samples(n: int) -> int:
    return 5000 if n <= 10 else max(5000, 200 * n)


def _default_weight(n: int) -> float:
    """Keep SEM variances stable on long chains (N=50+)."""
    return 2.0 if n <= 10 else 0.8


def _standardize_columns(X: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-variance columns (stabilizes Cov for large N)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    std = Xc.std(axis=0, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    return Xc / std


@dataclass(frozen=True)
class SyntheticBundle:
    """Shared SEM artifacts for synth harness and external benchmarks."""

    X: np.ndarray
    Cov: np.ndarray
    A_true: np.ndarray
    A_init: np.ndarray
    E_init: np.ndarray
    truth_edges: tuple[tuple[int, int], ...]
    n: int
    seed: int


def prepare_synthetic_bundle(
    *,
    n: int = N,
    n_samples: int | None = None,
    edge_prob: float | None = None,
    weight: float | None = None,
    seed: int = 0,
    target_edges: int | None = None,
) -> SyntheticBundle:
    """Build the same linear-DAG dataset used by :func:`run_phase2`."""
    rng = np.random.default_rng(seed)
    if n_samples is None:
        n_samples = _default_samples(n)
    if edge_prob is None:
        edge_prob = _default_edge_prob(n)
    if weight is None:
        weight = _default_weight(n)

    # Dedicated RNG stream for DAG construction so init A/E stay reproducible.
    dag_rng = np.random.default_rng(seed + 17_389)
    truth_edges = make_truth_edges(n, target_edges=target_edges, rng=dag_rng)
    X, _ = generate_linear_dag_samples(
        n_samples=n_samples, weight=weight, rng=rng, n=n, edges=truth_edges
    )
    if n > 10:
        X = _standardize_columns(X)
    Cov = empirical_covariance(X)
    A_true = truth_adjacency(n, edges=truth_edges)
    A_init = random_upper_tri_adjacency(n, p=edge_prob, rng=rng)
    E_init = rng.random(n)
    return SyntheticBundle(
        X=X,
        Cov=Cov,
        A_true=A_true,
        A_init=A_init,
        E_init=E_init,
        truth_edges=truth_edges,
        n=n,
        seed=seed,
    )


def run_phase2(
    *,
    n: int = N,
    n_samples: int | None = None,
    ticks: int = 100,
    edge_prob: float | None = None,
    weight: float | None = None,
    learning_rate: float = 0.1,
    threshold_death: float = 0.05,
    threshold_birth: float = 0.10,
    k: int = 3,
    seed: int = 0,
    target_edges: int | None = None,
    backend: str = "numpy",
) -> Phase2Result:
    """Generate data, init state, loop ``step`` (timed), return artifacts."""
    step_fn, device = _resolve_step(backend)
    bundle = prepare_synthetic_bundle(
        n=n,
        n_samples=n_samples,
        edge_prob=edge_prob,
        weight=weight,
        seed=seed,
        target_edges=target_edges,
    )
    X, Cov, A_true = bundle.X, bundle.Cov, bundle.A_true
    A = bundle.A_init.copy()
    E = bundle.E_init.copy()
    truth_edges = bundle.truth_edges

    metrics_init = edge_metrics(A, A_true)
    sk_init = skeleton_metrics(A, A_true)
    A_init = A.copy()
    E_init = E.copy()

    # Warmup JIT outside the timed window (no-op for NumPy).
    if backend == "jax" and ticks > 0:
        step_fn(
            A,
            E,
            Cov,
            learning_rate=learning_rate,
            threshold_death=threshold_death,
            threshold_birth=threshold_birth,
            k=k,
        )

    # Time only CA steps (metrics scored after the loop).
    t0 = time.perf_counter()
    for _ in range(ticks):
        A, E = step_fn(
            A,
            E,
            Cov,
            learning_rate=learning_rate,
            threshold_death=threshold_death,
            threshold_birth=threshold_birth,
            k=k,
        )
    elapsed = time.perf_counter() - t0

    metrics_final = edge_metrics(A, A_true)
    return Phase2Result(
        A_true=A_true,
        A_init=A_init,
        A_final=A,
        E_init=E_init,
        E_final=E,
        Cov=Cov,
        metrics_init=metrics_init,
        metrics_final=metrics_final,
        skeleton_init=sk_init,
        skeleton_final=skeleton_metrics(A, A_true),
        ticks=ticks,
        history=[metrics_init, metrics_final],
        seed=seed,
        n=n,
        truth_edges=truth_edges,
        elapsed_sec=elapsed,
        backend=backend,
        device=device,
    )


def report(result: Phase2Result) -> str:
    """Concise CLI summary: recovery metrics + throughput."""
    m0, m1 = result.metrics_init, result.metrics_final
    s1 = result.skeleton_final
    lines = [
        "=== Synthetic Bayesian CA ===",
        f"seed={result.seed}  N={result.n}  ticks={result.ticks}  "
        f"true_edges={len(result.truth_edges)}  "
        f"backend={result.backend}  device={result.device}",
        f"Ground-truth edges ({m1.true_edges}): "
        f"{format_edges(result.A_true, max_edges=24)}",
        f"Init directed:  TP={m0.tp} FP={m0.fp} FN={m0.fn}  "
        f"P={m0.precision:.2f} R={m0.recall:.2f} F1={m0.f1:.2f}  "
        f"(pred_edges={m0.pred_edges})",
        f"Final directed: TP={m1.tp} FP={m1.fp} FN={m1.fn}  "
        f"P={m1.precision:.2f} R={m1.recall:.2f} F1={m1.f1:.2f}  "
        f"(pred_edges={m1.pred_edges})",
        f"Final skeleton: TP={s1.tp} FP={s1.fp} FN={s1.fn}  "
        f"P={s1.precision:.2f} R={s1.recall:.2f} F1={s1.f1:.2f}",
        f"Wall time (step loop): {result.elapsed_sec:.4f}s  "
        f"ticks/s={result.ticks_per_sec:.2f}",
    ]
    if result.n <= 10:
        lines.append("True edge status (final):")
        for i, j in result.truth_edges:
            mark = "HIT" if result.A_final[i, j] > 0 else "MISS"
            lines.append(f"  {_edge_label(i, j, result.n)}: {mark}")
    return "\n".join(lines)


def report_multi(results: list[Phase2Result]) -> str:
    """Aggregate directed / skeleton metrics across seeds."""
    dir_f1 = np.array([r.metrics_final.f1 for r in results])
    dir_tp = np.array([r.metrics_final.tp for r in results], dtype=float)
    dir_fp = np.array([r.metrics_final.fp for r in results], dtype=float)
    dir_fn = np.array([r.metrics_final.fn for r in results], dtype=float)
    sk_f1 = np.array([r.skeleton_final.f1 for r in results])
    init_fp = np.array([r.metrics_init.fp for r in results], dtype=float)
    tps = np.array([r.ticks_per_sec for r in results])
    lines = [
        f"=== Multi-seed summary (n_runs={len(results)}, N={results[0].n}) ===",
        f"Directed final:  mean TP/FP/FN = "
        f"{dir_tp.mean():.2f}/{dir_fp.mean():.2f}/{dir_fn.mean():.2f}  "
        f"mean F1={dir_f1.mean():.2f} ± {dir_f1.std():.2f}",
        f"Skeleton final:  mean F1={sk_f1.mean():.2f} ± {sk_f1.std():.2f}",
        f"Mean init FP → final FP: {init_fp.mean():.1f} → {dir_fp.mean():.1f}",
        f"Ticks/s: mean={tps.mean():.2f}  min={tps.min():.2f}  max={tps.max():.2f}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        type=int,
        default=N,
        help="Number of nodes (10 = Phase 2; 50 = Phase 5 scale).",
    )
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--ticks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=int,
        default=0,
        help="If >0, run this many seeds (0..seeds-1) and print aggregate.",
    )
    parser.add_argument("--edge-prob", type=float, default=None)
    parser.add_argument(
        "--weight",
        type=float,
        default=None,
        help="SEM edge weight (default 2.0 for n=10, 0.8 for larger n).",
    )
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--threshold-death", type=float, default=0.05)
    parser.add_argument("--threshold-birth", type=float, default=0.10)
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Conditioning-set cap (Phase-2 default 3; step() API default remains 2).",
    )
    parser.add_argument(
        "--target-edges",
        type=int,
        default=None,
        help="True-edge count for n!=10 sparse DAGs (default ~62 for n=50).",
    )
    parser.add_argument(
        "--backend",
        choices=("numpy", "jax"),
        default="numpy",
        help="CA step backend (jax requires --extra jax or an accelerator wheel).",
    )
    args = parser.parse_args(argv)

    kwargs = dict(
        n=args.n,
        n_samples=args.samples,
        ticks=args.ticks,
        edge_prob=args.edge_prob,
        weight=args.weight,
        learning_rate=args.learning_rate,
        threshold_death=args.threshold_death,
        threshold_birth=args.threshold_birth,
        k=args.k,
        target_edges=args.target_edges,
        backend=args.backend,
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
