"""Benchmark JAX Bayesian CA vs gCastle PC / NOTEARS.

Default scale is **N=100** (same SEM helpers as :mod:`bayesian_synth`).

Install::

    uv sync --extra gcastle --extra jax
    # or: pip install gcastle

    # Colab T4 / NVIDIA (example):
    #   pip install -U "jax[cuda13]" gcastle torch
    #   # see JAX install docs if the CUDA extra name differs

Run::

    uv run python -m graphnet_automata.benchmark
    uv run python -m graphnet_automata.benchmark --n 100 --seed 0 --ticks 100
"""

from __future__ import annotations

import argparse
import inspect
import logging
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


def _p(*args, **kwargs) -> None:
    """Print with flush so device notes appear promptly under pipes/tee."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def print_device_checks() -> dict:
    """Print JAX / PyTorch device availability; return a small summary dict."""
    info: dict = {"torch_cuda": False, "jax_backend": "unavailable", "jax_devices": []}

    _p("=== Device checks ===")
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        info["torch_cuda"] = cuda
        _p(f"torch.cuda.is_available(): {cuda}")
        if cuda:
            _p(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
            _p(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
        else:
            _p("torch: CUDA not available (CPU)")
    except ImportError as exc:
        _p(f"torch: not installed ({exc})")

    try:
        import jax

        from graphnet_automata.jax_core import device_info

        info["jax_backend"] = jax.default_backend()
        info["jax_devices"] = [str(d) for d in jax.devices()]
        _p(f"jax.default_backend(): {info['jax_backend']}")
        _p(f"jax.devices(): {info['jax_devices']}")
        _p(f"jax_core.device_info(): {device_info()}")
    except ImportError as exc:
        _p(f"jax: not installed ({exc})")

    _p()
    return info


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


def _notears_kwargs(
    *,
    lambda1: float,
    w_threshold: float,
    max_iter: int,
    want_gpu: bool,
) -> tuple[dict, str]:
    """Build Notears constructor kwargs; honor ``device_type`` when supported."""
    _, Notears = _require_gcastle()
    kwargs: dict = {
        "lambda1": lambda1,
        "loss_type": "l2",
        "max_iter": max_iter,
        "w_threshold": w_threshold,
    }
    params = inspect.signature(Notears.__init__).parameters
    if "device_type" in params:
        device_type = "gpu" if want_gpu else "cpu"
        if want_gpu:
            _p("NOTEARS: passing device_type='gpu'")
        else:
            _p("NOTEARS: passing device_type='cpu' (CUDA unavailable)")
        kwargs["device_type"] = device_type
        note = f"lambda1={lambda1} w_threshold={w_threshold} max_iter={max_iter} device_type={device_type}"
        return kwargs, note

    # gcastle linear Notears is SciPy/CPU-only in current releases.
    if want_gpu:
        _p(
            "NOTEARS: castle.algorithms.Notears has no device_type in this gcastle "
            "build (linear SciPy solver) — falling back to CPU. "
            "NotearsNonlinear supports device_type='gpu' if you need a GPU path."
        )
    else:
        _p("NOTEARS: linear solver (CPU); no device_type in this gcastle build")
    note = (
        f"lambda1={lambda1} w_threshold={w_threshold} max_iter={max_iter} "
        "device=cpu(linear/scipy)"
    )
    return kwargs, note


def run_notears(
    X: np.ndarray,
    A_true: np.ndarray,
    *,
    lambda1: float = 0.1,
    w_threshold: float = 0.3,
    max_iter: int = 100,
    want_gpu: bool | None = None,
) -> BenchRow:
    _, Notears = _require_gcastle()
    if want_gpu is None:
        try:
            import torch

            want_gpu = bool(torch.cuda.is_available())
        except ImportError:
            want_gpu = False

    kwargs, note = _notears_kwargs(
        lambda1=lambda1,
        w_threshold=w_threshold,
        max_iter=max_iter,
        want_gpu=want_gpu,
    )
    t0 = time.perf_counter()
    nt = Notears(**kwargs)
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
        notes=note,
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
    A_pred = _as_binary_adj(np.asarray(A).copy())
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
    n: int = 100,
    seed: int = 0,
    ticks: int = 100,
    k: int = 3,
    n_samples: int | None = None,
) -> tuple[list[BenchRow], SyntheticBundle]:
    bundle = prepare_synthetic_bundle(n=n, seed=seed, n_samples=n_samples)
    try:
        import torch

        want_gpu = bool(torch.cuda.is_available())
    except ImportError:
        want_gpu = False

    rows = [
        run_pc(bundle.X, bundle.A_true),
        run_notears(bundle.X, bundle.A_true, want_gpu=want_gpu),
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
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of nodes (default 100; uses bayesian_synth SEM helpers).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ticks", type=int, default=100)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override sample count (default: synth helper, 200*n for n>10).",
    )
    args = parser.parse_args(argv)

    logging.getLogger("castle").setLevel(logging.WARNING)
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

    print_device_checks()
    _require_gcastle()
    rows, bundle = run_benchmark(
        n=args.n,
        seed=args.seed,
        ticks=args.ticks,
        k=args.k,
        n_samples=args.samples,
    )
    _p(
        f"N={bundle.n}  seed={bundle.seed}  samples={bundle.X.shape[0]}  "
        f"true_edges={len(bundle.truth_edges)}"
    )
    _p()
    _p(markdown_table(rows))
    _p()
    _p("Details:")
    for r in rows:
        _p(
            f"- {r.algorithm}: dir TP/FP/FN={r.directed_tp}/{r.directed_fp}/{r.directed_fn}  "
            f"skel TP/FP/FN={r.skeleton_tp}/{r.skeleton_fp}/{r.skeleton_fn}  "
            f"({r.notes})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
