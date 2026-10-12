"""Benchmark JAX Bayesian CA vs gCastle PC / GOLEM.

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

    # CPU / no-GPU smoke (import + GOLEM init only; no invented GPU timings):
    uv run python -m graphnet_automata.benchmark --smoke
"""

from __future__ import annotations

import argparse
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
        from castle.algorithms import GOLEM, PC
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise SystemExit(
            "gcastle is required for this benchmark.\n"
            "  uv sync --extra gcastle\n"
            "  # or: pip install gcastle\n"
            f"Import error: {exc}"
        ) from exc
    return PC, GOLEM


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def resolve_golem_device_type(*, want_gpu: bool | None = None) -> str:
    """Pick GOLEM ``device_type`` without crashing when CUDA is absent.

    gCastle's GOLEM raises ``ValueError`` if ``device_type='gpu'`` and
    ``torch.cuda.is_available()`` is False — so we only request GPU when CUDA
    is actually present.
    """
    if want_gpu is None:
        want_gpu = _cuda_available()
    if want_gpu and _cuda_available():
        _p("GOLEM: passing device_type='gpu'")
        return "gpu"
    if want_gpu and not _cuda_available():
        _p(
            "GOLEM: CUDA unavailable — falling back to device_type='cpu' "
            "(gCastle GOLEM raises ValueError if device_type='gpu' without CUDA)."
        )
        return "cpu"
    _p("GOLEM: passing device_type='cpu' (CUDA unavailable)")
    return "cpu"


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


def run_golem(
    X: np.ndarray,
    A_true: np.ndarray,
    *,
    lambda_1: float = 0.02,
    lambda_2: float = 5.0,
    num_iter: int = 100_000,
    graph_thres: float = 0.3,
    want_gpu: bool | None = None,
) -> BenchRow:
    """Run gCastle GOLEM (PyTorch); prefer GPU when CUDA is available."""
    _, GOLEM = _require_gcastle()
    device_type = resolve_golem_device_type(want_gpu=want_gpu)
    note = (
        f"lambda_1={lambda_1} lambda_2={lambda_2} num_iter={num_iter} "
        f"graph_thres={graph_thres} device_type={device_type}"
    )
    t0 = time.perf_counter()
    model = GOLEM(
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        num_iter=num_iter,
        graph_thres=graph_thres,
        device_type=device_type,
    )
    model.learn(X)
    elapsed = time.perf_counter() - t0
    A_pred = _as_binary_adj(model.causal_matrix)
    d, s = _score(A_pred, A_true)
    return BenchRow(
        algorithm="GOLEM (gcastle)",
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


def smoke_golem_init() -> None:
    """Import/init GOLEM without ``learn`` — for CPU VMs / CI smoke tests."""
    _, GOLEM = _require_gcastle()
    device_type = resolve_golem_device_type()
    model = GOLEM(device_type=device_type, num_iter=10, checkpoint_iter=5)
    _p(f"GOLEM smoke init OK: device={model.device} device_type={model.device_type}")

    # Document gCastle's hard fail if callers pass gpu blindly.
    if not _cuda_available():
        try:
            GOLEM(device_type="gpu", num_iter=1)
            _p("GOLEM(device_type='gpu') unexpectedly succeeded without CUDA")
        except ValueError as exc:
            _p(f"GOLEM(device_type='gpu') without CUDA raises ValueError (expected): {exc}")


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
    golem_iters: int = 100_000,
) -> tuple[list[BenchRow], SyntheticBundle]:
    bundle = prepare_synthetic_bundle(n=n, seed=seed, n_samples=n_samples)
    want_gpu = _cuda_available()

    rows = [
        run_pc(bundle.X, bundle.A_true),
        run_golem(bundle.X, bundle.A_true, want_gpu=want_gpu, num_iter=golem_iters),
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
    parser.add_argument(
        "--golem-iters",
        type=int,
        default=100_000,
        help="GOLEM num_iter (default 100000, matching gCastle).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Device checks + GOLEM import/init only (no full learn / no timings).",
    )
    args = parser.parse_args(argv)

    logging.getLogger("castle").setLevel(logging.WARNING)
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

    print_device_checks()
    _require_gcastle()

    if args.smoke:
        smoke_golem_init()
        _p()
        _p(
            "Smoke complete (no GOLEM.learn / no F1 table). "
            "For Colab T4 full numbers:"
        )
        _p('  pip install -U "jax[cuda13]" gcastle torch')
        _p("  python -m graphnet_automata.benchmark --n 100")
        return 0

    rows, bundle = run_benchmark(
        n=args.n,
        seed=args.seed,
        ticks=args.ticks,
        k=args.k,
        n_samples=args.samples,
        golem_iters=args.golem_iters,
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
