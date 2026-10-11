"""Smoke tests for the JAX Bayesian CA port (optional ``jax`` extra)."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from graphnet_automata.jax_core import device_info, step, warmup


def test_warmup_compiles() -> None:
    info = warmup(n=6, k=2)
    assert "jax" in info
    assert device_info()


def test_jax_step_shapes_and_downhill() -> None:
    n = 8
    rng = np.random.default_rng(0)
    A = np.triu(rng.random((n, n)) < 0.3, k=1).astype(np.float64)
    E = rng.random(n)
    X = rng.normal(size=(200, n))
    Cov = np.cov(X, rowvar=False) + 0.1 * np.eye(n)

    A1, E1 = step(jnp.asarray(A), jnp.asarray(E), jnp.asarray(Cov), k=2)
    A1_np, E1_np = np.asarray(A1), np.asarray(E1)
    assert A1_np.shape == (n, n)
    assert E1_np.shape == (n,)
    assert set(np.unique(A1_np)).issubset({0.0, 1.0})
    assert np.all(np.diag(A1_np) == 0.0)
    src, dst = np.nonzero(A1_np)
    if src.size:
        assert np.all(E1_np[src] > E1_np[dst])
