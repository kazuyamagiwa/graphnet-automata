"""Bayesian CA update: DAG adjacency evolves via partial-correlation CMI.

NumPy prototype. Core state is binary adjacency ``A``, node elevations ``E``,
and a fixed global covariance ``Cov``. Each tick applies gravity (downhill-only
edges + elevation drift), a V-structure gravity well that sinks colliders,
then death / birth from batched partial correlations — no sequential ``N×N``
edge loops.

Birth evaluates all absent undirected pairs (no path-2 filter); elevation
chooses edge direction after the ``|rho|`` threshold is met. Death/birth
condition on the top-``k`` neighbors ranked by absolute covariance vs the pair.
"""

from __future__ import annotations

import numpy as np

# Small ridge for sub-covariance inversion (linear-Gaussian precision).
_RIDGE = 1e-8


def _undirected_neighbors(A: np.ndarray) -> np.ndarray:
    """Binary undirected adjacency with zero diagonal."""
    U = ((A + A.T) > 0).astype(np.float64)
    np.fill_diagonal(U, 0.0)
    return U


def _topk_by_score(
    mask: np.ndarray, scores: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Take up to ``k`` masked columns per row with highest ``scores``.

    Invalid (``mask`` False) entries are ignored. Ties break toward the lower
    column index. ``Z`` is left-packed with ``-1`` padding.

    Parameters
    ----------
    mask:
        Boolean array of shape ``(M, N)`` — candidate pool per row.
    scores:
        Float array of shape ``(M, N)`` — ranking scores (higher is better).
    k:
        Maximum number of indices to keep per row.

    Returns
    -------
    Z, counts
        ``Z`` has shape ``(M, k)`` with ``-1`` padding; ``counts`` is ``(M,)``.
    """
    M, N = mask.shape
    if k <= 0:
        return np.full((M, 0), -1, dtype=np.intp), np.zeros(M, dtype=np.intp)

    # Invalid → -inf so they sort after all real scores under descending order.
    ranked = np.where(mask, scores, -np.inf)
    # Ascending sort key: lower is better → high score first; +index for ties.
    sort_key = -ranked + np.arange(N, dtype=np.float64) * 1e-12
    order = np.argsort(sort_key, axis=1)
    top = order[:, :k]
    rows = np.arange(M)[:, None]
    valid = mask[rows, top]
    Z = np.where(valid, top, -1).astype(np.intp)
    counts = valid.sum(axis=1).astype(np.intp)
    return Z, counts


def _batched_partial_corr(
    Cov: np.ndarray,
    I: np.ndarray,
    J: np.ndarray,
    Z: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Partial correlation of ``(I[m], J[m])`` given ``Z[m, :counts[m]]``.

    Groups edges by conditioning-set size and inverts stacked sub-covariances
    with a single ``numpy.linalg.inv`` call per group.
    """
    M = I.shape[0]
    rho = np.zeros(M, dtype=np.float64)
    if M == 0:
        return rho

    k = Z.shape[1] if Z.ndim == 2 else 0
    for n_z in range(0, k + 1):
        idx = np.flatnonzero(counts == n_z)
        if idx.size == 0:
            continue
        parts = [I[idx], J[idx]]
        for t in range(n_z):
            parts.append(Z[idx, t])
        nodes = np.column_stack(parts)  # (m, 2+n_z)
        # Advanced indexing: Cov[nodes[a], nodes[b]] for each row.
        sub = Cov[nodes[:, :, None], nodes[:, None, :]].astype(np.float64)
        eye = np.eye(sub.shape[-1], dtype=np.float64)
        sub = sub + _RIDGE * eye
        P = np.linalg.inv(sub)
        denom = np.sqrt(np.maximum(P[:, 0, 0] * P[:, 1, 1], 0.0))
        rho[idx] = np.where(denom > 0.0, -P[:, 0, 1] / denom, 0.0)
    return rho


def _union_neighbor_sets(
    U: np.ndarray,
    I: np.ndarray,
    J: np.ndarray,
    k: int,
    Cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` undirected neighbors of ``i`` or ``j``, ranked by |Cov|.

    Pool = neighbors of ``i`` or ``j`` (excluding endpoints). Score for
    candidate ``m`` is ``max(|Cov[m,i]|, |Cov[m,j]|)`` so conditioning prefers
    the strongest known confounders rather than lowest index.
    """
    M = I.shape[0]
    if M == 0:
        return np.full((0, k), -1, dtype=np.intp), np.zeros(0, dtype=np.intp)
    # U[I] → (M, N); boolean OR of neighbor rows.
    mask = (U[I] > 0) | (U[J] > 0)
    mask[np.arange(M), I] = False
    mask[np.arange(M), J] = False
    # scores[m, n] = max(|Cov[n, I[m]]|, |Cov[n, J[m]]|)
    score_i = np.abs(Cov[:, I].T)
    score_j = np.abs(Cov[:, J].T)
    scores = np.maximum(score_i, score_j)
    return _topk_by_score(mask, scores, k)


def _vstructure_gravity_well(
    A: np.ndarray,
    E: np.ndarray,
    Cov: np.ndarray,
    *,
    learning_rate: float,
    threshold_death: float,
) -> np.ndarray:
    """Sink elevations of collider nodes implied by V-structure pattern.

    Absent undirected pairs ``(i, j)`` that share neighbor(s) via ``U @ U`` and
    are marginally independent (``|rho| < threshold_death`` with empty ``Z``)
    mark each shared neighbor ``k`` as a collider. Those elevations are lowered
    by ``learning_rate * 2`` (once per tick if flagged), then clipped to
    ``[0, 1]``.
    """
    U = _undirected_neighbors(A)
    undirected_absent = (A + A.T) == 0.0
    np.fill_diagonal(undirected_absent, False)
    shared_counts = U @ U
    np.fill_diagonal(shared_counts, 0.0)
    pair_mask = np.triu(undirected_absent & (shared_counts > 0), k=1)
    Pi, Pj = np.nonzero(pair_mask)
    if Pi.size == 0:
        return E

    # Marginal / empty-Z partial correlation (batched 2×2 precision).
    Z_empty = np.zeros((Pi.size, 0), dtype=np.intp)
    counts0 = np.zeros(Pi.size, dtype=np.intp)
    rho = _batched_partial_corr(Cov, Pi, Pj, Z_empty, counts0)
    independent = np.abs(rho) < threshold_death
    if not np.any(independent):
        return E

    Ii = Pi[independent]
    Ij = Pj[independent]
    shared = (U[Ii] > 0) & (U[Ij] > 0)  # (M_ind, N)
    collider = shared.any(axis=0)  # unique k flagged this tick
    E = E.copy()
    E[collider] -= learning_rate * 2.0
    return np.clip(E, 0.0, 1.0)


def step(
    A: np.ndarray,
    E: np.ndarray,
    Cov: np.ndarray,
    *,
    learning_rate: float = 0.1,
    threshold_death: float = 0.05,
    threshold_birth: float = 0.15,
    k: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """One Bayesian CA tick: gravity, V-well, then edge death / birth.

    Parameters
    ----------
    A:
        Binary adjacency ``(N, N)``; ``A[i, j] = 1`` means directed edge ``i→j``.
    E:
        Node elevations ``(N,)``, intended in ``[0, 1]``.
    Cov:
        Global covariance ``(N, N)`` of the continuous observations.
    learning_rate:
        Step size for elevation drift toward parent/child midpoint.
    threshold_death:
        Kill existing edges with ``|rho|`` below this value.
    threshold_birth:
        Birth candidate edges with ``|rho|`` above this value.
    k:
        Cap on conditioning-set size ``|Z|`` (default 2). Death/birth pick the
        top-``k`` neighbors by ``max(|Cov[m,i]|, |Cov[m,j]|)``.

    Returns
    -------
    A_new, E_new
        Updated binary adjacency and elevations (copies; inputs are not mutated).
    """
    A = np.asarray(A, dtype=np.float64).copy()
    E = np.asarray(E, dtype=np.float64).copy().reshape(-1)
    Cov = np.asarray(Cov, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square adjacency matrix")
    N = A.shape[0]
    if E.shape != (N,):
        raise ValueError(f"E must have shape ({N},), got {E.shape}")
    if Cov.shape != (N, N):
        raise ValueError(f"Cov must have shape ({N}, {N}), got {Cov.shape}")
    if k < 0:
        raise ValueError("k must be non-negative")

    # --- 1. Gravity: downhill-only edges, then elevation drift ---------------
    gravity_ok = E[:, None] > E[None, :]
    np.fill_diagonal(gravity_ok, False)
    A *= gravity_ok

    parent_count = A.sum(axis=0)
    child_count = A.sum(axis=1)
    parent_sum = A.T @ E
    child_sum = A @ E
    # Isolated side: treat missing mean as the node's own elevation (no pull).
    mean_parents = np.divide(
        parent_sum, parent_count, out=E.copy(), where=parent_count > 0
    )
    mean_children = np.divide(
        child_sum, child_count, out=E.copy(), where=child_count > 0
    )
    E = E + learning_rate * ((mean_parents + mean_children) * 0.5 - E)
    E = np.clip(E, 0.0, 1.0)

    # --- 2. V-structure gravity well (before death / birth edge updates) -----
    E = _vstructure_gravity_well(
        A,
        E,
        Cov,
        learning_rate=learning_rate,
        threshold_death=threshold_death,
    )

    # Neighbor structure from the post-gravity DAG (undirected for Z).
    U = _undirected_neighbors(A)

    # --- 3. Death: weak partial correlation given Cov-ranked neighbor Z -----
    I, J = np.nonzero(A > 0)
    if I.size:
        Z, counts = _union_neighbor_sets(U, I, J, k, Cov)
        rho = _batched_partial_corr(Cov, I, J, Z, counts)
        kill = np.abs(rho) < threshold_death
        if np.any(kill):
            A[I[kill], J[kill]] = 0.0
            U = _undirected_neighbors(A)

    # --- 4. Birth: all absent undirected pairs; elevation picks direction ----
    # Unordered pairs with no edge either way (batched via upper triangle).
    undirected_absent = (A + A.T) == 0.0
    np.fill_diagonal(undirected_absent, False)
    Bi, Bj = np.nonzero(np.triu(undirected_absent, k=1))
    if Bi.size:
        Z, counts = _union_neighbor_sets(U, Bi, Bj, k, Cov)
        rho = _batched_partial_corr(Cov, Bi, Bj, Z, counts)
        strong = np.abs(rho) > threshold_birth
        # Direction from elevations after the rho gate (equal E → no-op).
        forward = strong & (E[Bi] > E[Bj])  # i → j
        backward = strong & (E[Bj] > E[Bi])  # j → i
        if np.any(forward):
            A[Bi[forward], Bj[forward]] = 1.0
        if np.any(backward):
            A[Bj[backward], Bi[backward]] = 1.0

    # Binary adjacency; clear flat/uphill edges after collider sinks reshape E.
    A = (A > 0).astype(np.float64)
    A *= E[:, None] > E[None, :]
    np.fill_diagonal(A, 0.0)
    return A, E
