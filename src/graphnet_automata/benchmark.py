"""Benchmark JAX Bayesian CA vs gCastle PC / NOTEARS on the N=50 SEM.

Reuses :func:`bayesian_synth.prepare_synthetic_bundle` so the linear DAG and
samples match the Phase-5 / Phase-6 harness.

Install::

    uv sync --extra gcastle --extra jax
    # or: pip install gcastle

Run::

    uv run python -m graphnet_automata.benchmark
    uv run python -m graphnet_automata.benchmark --n 50 --seed 0 --ticks 100
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from graphnet_automata.bayesian_synth import (
    SyntheticBundle,
    edge_metrics,
    prepare_synthetic_bundle,
    skeleton_metrics,
)


@dataclass(frozen=True)
class BenchRow:
    algorithm: str
    skeleton_f1: float
    directed_f1: float
    wall_time_s: float
    directed_tp: int
    directed_fp: int
    directed_fn: int
    skeleton_tp: int
    skeleton_fp: int
    skeleton_fn: int
    notes: str = ""


def _require_gcastle():
    try:
        from castle.algorithms import Notears, PC
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise SystemExit(
            "gcastle is required for this benchmark.\n"
            "  uv sync --extra gcastle\n"
            "  # or: pip install gcastle\n"
            f"Import error: {exc}"
        ) from exc
    return PC, Notears


def _as_binary_adj(mat) -> np.ndarray:
    # JAX device arrays often yield read-only NumPy views; force a writeable copy.
    A = np.asarray(mat, dtype=np.float64).copy()
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expected square adjacency, got shape {A.shape}")
    np.fill_diagonal(A, 0.0)
    return (A > 0).astype(np.float64)


def _score(A_pred: np.ndarray, A_true: np.ndarray) -> tuple:
    d = edge_metrics(A_pred, A_true)
    s = skeleton_metrics(A_pred, A_true)
    return d, s


def run_pc(X: np.ndarray, A_true: np.ndarray, *, alpha: float = 0.05) -> BenchRow:
    PC, _ = _require_gcastle()
    t0 = time.perf_counter()
    pc = PC(variant="stable", alpha=alpha, ci_test="fisherz")
    pc.learn(X)
    elapsed = time.perf_counter() - t0
    A_pred = _as_binary_adj(pc.causal_matrix)
    d, s = _score(A_pred, A_true)
    return BenchRow(
        algorithm="PC (gcastle)",
        skeleton_f1=s.f1,
        directed_f1=d.f1,
        wall_time_s=elapsed,
        directed_tp=d.tp,
        directed_fp=d.fp,
        directed_fn=d.fn,
        skeleton_tp=s.tp,
        skeleton_fp=s.fp,
        skeleton_fn=s.fn,
        notes=f"variant=stable alpha={alpha} fisherz",
    )


def run_notears(
    X: np.ndarray,
    A_true: np.ndarray,
    *,
    lambda1: float = 0.1,
    w_threshold: float = 0.3,
    max_iter: int = 100,
) -> BenchRow:
    _, Notears = _require_gcastle()
    t0 = time.perf_counter()
    nt = Notears(lambda1=lambda1, loss_type="l2", max_iter=max_iter, w_threshold=w_threshold)
    nt.learn(X)
    elapsed = time.perf_counter() - t0
    A_pred = _as_binary_adj(nt.causal_matrix)
    d, s = _score(A_pred, A_true)
    return BenchRow(
        algorithm="NOTEARS (gcastle)",
        skeleton_f1=s.f1,
        directed_f1=d.f1,
        wall_time_s=elapsed,
        directed_tp=d.tp,
        directed_fp=d.fp,
        directed_fn=d.fn,
        skeleton_tp=s.tp,
        skeleton_fp=s.fp,
        skeleton_fn=s.fn,
        notes=f"lambda1={lambda1} w_threshold={w_threshold} max_iter={max_iter}",
    )


def run_jax_ca(
    Cov: np.ndarray,
    A_init: np.ndarray,
    E_init: np.ndarray,
    A_true: np.ndarray,
    *,
    ticks: int = 100,
    learning_rate: float = 0.1,
    threshold_death: float = 0.05,
    threshold_birth: float = 0.10,
    k: int = 3,
) -> BenchRow:
    try:
        import jax.numpy as jnp

        from graphnet_automata.jax_core import device_info, step as jax_step
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "JAX is required for the CA row of this benchmark.\n"
            "  uv sync --extra jax\n"
            f"Import error: {exc}"
        ) from exc

    A = jnp.asarray(A_init)
    E = jnp.asarray(E_init)
    Cov_j = jnp.asarray(Cov)
    # Warmup compile outside the timed window.
    jax_step(
        A,
        E,
        Cov_j,
        learning_rate=learning_rate,
        threshold_death=threshold_death,
        threshold_birth=threshold_birth,
        k=k,
    )
    t0 = time.perf_counter()
    for _ in range(ticks):
        A, E = jax_step(
            A,
            E,
            Cov_j,
            learning_rate=learning_rate,
            threshold_death=threshold_death,
            threshold_birth=threshold_birth,
            k=k,
        )
    elapsed = time.perf_counter() - t0
    A_pred = _as_binary_adj(np.asarray(A))
    d, s = _score(A_pred, A_true)
    return BenchRow(
        algorithm="Bayesian CA (JAX)",
        skeleton_f1=s.f1,
        directed_f1=d.f1,
        wall_time_s=elapsed,
        directed_tp=d.tp,
        directed_fp=d.fp,
        directed_fn=d.fn,
        skeleton_tp=s.tp,
        skeleton_fp=s.fp,
        skeleton_fn=s.fn,
        notes=f"ticks={ticks} k={k} {device_info()}",
    )


def markdown_table(rows: list[BenchRow]) -> str:
    lines = [
        "| Algorithm | Skeleton F1 | Directed F1 | Wall Time (s) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r.algorithm} | {r.skeleton_f1:.3f} | {r.directed_f1:.3f} | "
            f"{r.wall_time_s:.3f} |"
        )
    return "\n".join(lines)


def run_benchmark(
    *,
    n: int = 50,
    seed: int = 0,
    ticks: int = 100,
    k: int = 3,
) -> tuple[list[BenchRow], SyntheticBundle]:
    bundle = prepare_synthetic_bundle(n=n, seed=seed)
    rows = [
        run_pc(bundle.X, bundle.A_true),
        run_notears(bundle.X, bundle.A_true),
        run_jax_ca(
            bundle.Cov,
            bundle.A_init,
            bundle.E_init,
            bundle.A_true,
            ticks=ticks,
            k=k,
        ),
    ]
    return rows, bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ticks", type=int, default=100)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args(argv)

    import logging

    logging.getLogger("castle").setLevel(logging.WARNING)

    _require_gcastle()
    rows, bundle = run_benchmark(
        n=args.n, seed=args.seed, ticks=args.ticks, k=args.k
    )
    print(
        f"N={bundle.n}  seed={bundle.seed}  samples={bundle.X.shape[0]}  "
        f"true_edges={len(bundle.truth_edges)}"
    )
    print()
    print(markdown_table(rows))
    print()
    print("Details:")
    for r in rows:
        print(
            f"- {r.algorithm}: dir TP/FP/FN={r.directed_tp}/{r.directed_fp}/{r.directed_fn}  "
            f"skel TP/FP/FN={r.skeleton_tp}/{r.skeleton_fp}/{r.skeleton_fn}  "
            f"({r.notes})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
