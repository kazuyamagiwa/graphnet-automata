"""Smoke tests for the Phase-1 Bayesian CA ``step``."""

import numpy as np
import pytest

from graphnet_automata.bayesian import step


def _spd_cov(n: int, rng: np.random.Generator) -> np.ndarray:
    X = rng.normal(size=(n + 4, n))
    return (X.T @ X) / X.shape[0] + 0.1 * np.eye(n)


def test_step_shapes_and_binary() -> None:
    rng = np.random.default_rng(0)
    n = 6
    A = (rng.random((n, n)) < 0.3).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    E = rng.random(n)
    Cov = _spd_cov(n, rng)

    A2, E2 = step(A, E, Cov)
    assert A2.shape == (n, n)
    assert E2.shape == (n,)
    assert set(np.unique(A2)).issubset({0.0, 1.0})
    assert np.all(np.diag(A2) == 0.0)
    assert np.all((E2 >= 0.0) & (E2 <= 1.0))


def test_step_preserves_downhill_dag() -> None:
    rng = np.random.default_rng(1)
    n = 8
    E = np.linspace(1.0, 0.0, n)
    # Upper-tri edges point high→low index; with decreasing E that is downhill.
    A = np.triu(np.ones((n, n)), k=1)
    Cov = _spd_cov(n, rng)

    A2, E2 = step(A, E, Cov, learning_rate=0.05)
    src, dst = np.nonzero(A2)
    assert np.all(E2[src] > E2[dst])


def test_step_kwargs_defaults_accepted() -> None:
    n = 4
    A = np.zeros((n, n))
    A[0, 1] = A[1, 2] = 1.0
    E = np.array([0.9, 0.5, 0.2, 0.1])
    Cov = np.eye(n)
    A2, E2 = step(
        A,
        E,
        Cov,
        learning_rate=0.1,
        threshold_death=0.05,
        threshold_birth=0.15,
        k=2,
    )
    assert A2.shape == (n, n)
    assert E2.shape == (n,)


def test_step_rejects_bad_shapes() -> None:
    A = np.eye(3)
    E = np.zeros(2)
    Cov = np.eye(3)
    with pytest.raises(ValueError):
        step(A, E, Cov)
