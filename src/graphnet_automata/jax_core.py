"""JAX port of the Bayesian CA ``step`` (functional + ``@jax.jit``).

Cross-platform: uses only ``jax`` / ``jax.numpy``. Accelerator selection is
left to the installed JAX wheel (CPU, CUDA, or Metal/MPS) — no hardcoded
backend or Mac-specific branches in this module.

Install (pick one accelerator wheel for your machine)::

    uv sync --extra jax          # CPU jax/jaxlib
    # Apple Silicon:  pip/uv install jax-mps
    # NVIDIA / Colab: pip/uv install "jax[cuda]"  (see JAX install docs)

Smoke::

    from graphnet_automata.jax_core import step, device_info
    A1, E1 = step(A, E, Cov)  # jit-compiled on first call
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

# Match the NumPy prototype's float64 path (partial-corr precision).
jax.config.update("jax_enable_x64", True)

_RIDGE = 1e-8


def device_info() -> str:
    """Short description of the default JAX backend / device."""
    devs = jax.devices()
    if not devs:
        return "jax: (no devices)"
    d0 = devs[0]
    return f"jax backend={jax.default_backend()} device={d0.platform}:{d0.device_kind}"


def _undirected_neighbors(A: jnp.ndarray) -> jnp.ndarray:
    """Binary undirected adjacency with zero diagonal."""
    n = A.shape[0]
    U = ((A + A.T) > 0).astype(A.dtype)
    return U.at[jnp.diag_indices(n)].set(0)


def _pairwise_topk_Z(
    U: jnp.ndarray, Cov: jnp.ndarray, k: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Top-``k`` union neighbors for every pair ``(i, j)``, ranked by |Cov|.

    Returns
    -------
    Z, valid
        ``Z`` shape ``(N, N, k)``; ``valid`` same shape (bool).
    """
    n = U.shape[0]
    idx = jnp.arange(n)
    # pool[i, j, m] = neighbor of i or j, excluding endpoints
    pool = (U[:, None, :] > 0) | (U[None, :, :] > 0)
    pool = pool & (idx[None, None, :] != idx[:, None, None])
    pool = pool & (idx[None, None, :] != idx[None, :, None])

    # scores[i, j, m] = max(|Cov[m, i]|, |Cov[m, j]|)
    score_i = jnp.abs(Cov.T)[:, None, :]  # (i, 1, m)
    score_j = jnp.abs(Cov.T)[None, :, :]  # (1, j, m)
    scores = jnp.maximum(score_i, score_j)

    ranked = jnp.where(pool, scores, -jnp.inf)
    sort_key = -ranked + idx.astype(ranked.dtype) * jnp.asarray(1e-12, ranked.dtype)
    order = jnp.argsort(sort_key, axis=-1)
    Z = order[..., :k]
    # valid[i,j,t] = pool[i,j,Z[i,j,t]]
    ii = idx[:, None, None]
    jj = idx[None, :, None]
    valid = pool[ii, jj, Z]
    return Z, valid


def _pairwise_partial_corr(
    Cov: jnp.ndarray, Z: jnp.ndarray, valid: jnp.ndarray, k: int
) -> jnp.ndarray:
    """``rho[i, j]`` for all pairs, conditioning on padded top-``k`` ``Z``.

    Invalid ``Z`` slots map to independent dummy variables appended to ``Cov``
    so the invert size stays fixed under JIT (conditioning on independents is a
    no-op relative to a smaller ``Z``).
    """
    n = Cov.shape[0]
    if k == 0:
        ii = jnp.arange(n)[:, None]
        jj = jnp.arange(n)[None, :]
        nodes = jnp.stack(
            [jnp.broadcast_to(ii, (n, n)), jnp.broadcast_to(jj, (n, n))],
            axis=-1,
        )
        sub = Cov[nodes[..., :, None], nodes[..., None, :]]
    else:
        # Augment Cov with k independent unit dummies.
        cov_aug = jnp.zeros((n + k, n + k), dtype=Cov.dtype)
        cov_aug = cov_aug.at[:n, :n].set(Cov)
        cov_aug = cov_aug.at[n:, n:].set(jnp.eye(k, dtype=Cov.dtype))
        dummy = n + jnp.arange(k)
        z_pad = jnp.where(valid, Z, dummy)
        ii = jnp.arange(n)[:, None, None]
        jj = jnp.arange(n)[None, :, None]
        nodes = jnp.concatenate(
            [
                jnp.broadcast_to(ii, (n, n, 1)),
                jnp.broadcast_to(jj, (n, n, 1)),
                z_pad,
            ],
            axis=-1,
        )
        sub = cov_aug[nodes[..., :, None], nodes[..., None, :]]

    eye = jnp.eye(sub.shape[-1], dtype=Cov.dtype)
    scale = jnp.maximum(jnp.mean(jnp.diagonal(sub, axis1=-2, axis2=-1), axis=-1), 1e-8)
    sub = sub + (_RIDGE + 1e-6 * scale)[..., None, None] * eye
    precision = jnp.linalg.inv(sub)
    denom = jnp.sqrt(jnp.maximum(precision[..., 0, 0] * precision[..., 1, 1], 0.0))
    rho = jnp.where(denom > 0.0, -precision[..., 0, 1] / denom, 0.0)
    return rho


def _empty_z_partial_corr(Cov: jnp.ndarray) -> jnp.ndarray:
    """Marginal / empty-``Z`` partial correlation for all pairs (2×2)."""
    n = Cov.shape[0]
    z = jnp.zeros((n, n, 0), dtype=jnp.int32)
    valid = jnp.zeros((n, n, 0), dtype=bool)
    return _pairwise_partial_corr(Cov, z, valid, k=0)


def _vstructure_gravity_well(
    A: jnp.ndarray,
    E: jnp.ndarray,
    Cov: jnp.ndarray,
    learning_rate: float,
    threshold_death: float,
) -> jnp.ndarray:
    """Sink elevations of collider nodes (functional)."""
    n = A.shape[0]
    U = _undirected_neighbors(A)
    undirected_absent = (A + A.T) == 0
    undirected_absent = undirected_absent.at[jnp.diag_indices(n)].set(False)
    shared_counts = U @ U
    shared_counts = shared_counts.at[jnp.diag_indices(n)].set(0)
    # Upper-triangular candidate pairs with shared neighbors.
    pair_mask = jnp.triu(undirected_absent & (shared_counts > 0), k=1)

    rho = _empty_z_partial_corr(Cov)
    independent = (jnp.abs(rho) < threshold_death) & pair_mask

    # shared[i,j,m] for independent pairs → collider flags on m
    shared = (U[:, None, :] > 0) & (U[None, :, :] > 0)
    # Zero endpoints in shared mask
    idx = jnp.arange(n)
    shared = shared & (idx[None, None, :] != idx[:, None, None])
    shared = shared & (idx[None, None, :] != idx[None, :, None])
    collider = jnp.any(shared & independent[..., None], axis=(0, 1))
    E_new = jnp.where(collider, E - learning_rate * 2.0, E)
    return jnp.clip(E_new, 0.0, 1.0)


@partial(jax.jit, static_argnames=("k",))
def step(
    A: jnp.ndarray,
    E: jnp.ndarray,
    Cov: jnp.ndarray,
    learning_rate: float = 0.1,
    threshold_death: float = 0.05,
    threshold_birth: float = 0.15,
    k: int = 2,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One Bayesian CA tick (JAX): gravity, V-well, death, elev-orient birth.

    Pure functional; wrapped in ``@jax.jit``. ``k`` is static (shapes).
    """
    A = jnp.asarray(A, dtype=jnp.float64)
    E = jnp.asarray(E, dtype=jnp.float64).reshape(-1)
    Cov = jnp.asarray(Cov, dtype=jnp.float64)
    n = A.shape[0]

    # --- 1. Gravity: downhill-only edges, then elevation drift ---------------
    gravity_ok = E[:, None] > E[None, :]
    gravity_ok = gravity_ok.at[jnp.diag_indices(n)].set(False)
    A = A * gravity_ok

    parent_count = A.sum(axis=0)
    child_count = A.sum(axis=1)
    parent_sum = A.T @ E
    child_sum = A @ E
    mean_parents = jnp.where(parent_count > 0, parent_sum / parent_count, E)
    mean_children = jnp.where(child_count > 0, child_sum / child_count, E)
    E = E + learning_rate * ((mean_parents + mean_children) * 0.5 - E)
    E = jnp.clip(E, 0.0, 1.0)

    # --- 2. V-structure gravity well -----------------------------------------
    E = _vstructure_gravity_well(A, E, Cov, learning_rate, threshold_death)

    U = _undirected_neighbors(A)

    # Pairwise Cov-ranked Z and partial correlations (fixed N×N×k shapes).
    Z, valid = _pairwise_topk_Z(U, Cov, k)
    rho = _pairwise_partial_corr(Cov, Z, valid, k)

    # --- 3. Death ------------------------------------------------------------
    existing = A > 0
    kill = existing & (jnp.abs(rho) < threshold_death)
    A = jnp.where(kill, 0.0, A)
    U = _undirected_neighbors(A)
    Z, valid = _pairwise_topk_Z(U, Cov, k)
    rho = _pairwise_partial_corr(Cov, Z, valid, k)

    # --- 4. Birth: all absent undirected pairs; elevation picks direction ----
    undirected_absent = (A + A.T) == 0
    undirected_absent = undirected_absent.at[jnp.diag_indices(n)].set(False)
    # Evaluate unordered pairs once via upper triangle, then orient.
    candidates = jnp.triu(undirected_absent, k=1)
    strong = candidates & (jnp.abs(rho) > threshold_birth)
    forward = strong & (E[:, None] > E[None, :])
    backward = strong & (E[None, :] > E[:, None])
    # backward[i,j] means E[j] > E[i] on upper-tri (i<j) ⇒ birth j→i
    A = jnp.where(forward, 1.0, A)
    A = jnp.where(backward.T, 1.0, A)  # place j→i at (j,i)

    # Binary + downhill clear (after collider sinks may flatten E).
    A = (A > 0).astype(A.dtype)
    A = A * (E[:, None] > E[None, :])
    A = A.at[jnp.diag_indices(n)].set(0.0)
    return A, E


def warmup(
    n: int = 8,
    *,
    k: int = 2,
) -> str:
    """Compile ``step`` on a tiny problem and return device info."""
    A = jnp.triu(jnp.ones((n, n), dtype=jnp.float64), k=1) * 0.0
    E = jnp.linspace(1.0, 0.0, n, dtype=jnp.float64)
    Cov = jnp.eye(n, dtype=jnp.float64)
    # First call triggers JIT compile.
    step(A, E, Cov, k=k)
    return device_info()
