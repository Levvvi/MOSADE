"""Weight vector generation and neighborhood management for decomposition.

Implements the Das-Dennis method for uniform weight vector generation
and neighborhood computation by Euclidean distance in weight space.
"""

from __future__ import annotations

import numpy as np


def das_dennis(n_partitions: int, n_obj: int) -> np.ndarray:
    """Generate uniformly distributed weight vectors on the unit simplex.

    Uses the Das-Dennis systematic approach.  The number of vectors
    produced is C(n_partitions + n_obj - 1, n_obj - 1).

    Parameters
    ----------
    n_partitions : int
        Number of divisions along each objective axis (H).
    n_obj : int
        Number of objectives (M).

    Returns
    -------
    ndarray, shape (N, n_obj)
        Weight vectors that sum to 1.
    """
    # Generate all combinations of n_obj non-negative integers that sum to n_partitions
    def _recurse(m: int, h: int) -> list[list[int]]:
        if m == 1:
            return [[h]]
        result = []
        for i in range(h + 1):
            for tail in _recurse(m - 1, h - i):
                result.append([i] + tail)
        return result

    raw = np.array(_recurse(n_obj, n_partitions), dtype=float)
    return raw / n_partitions


def compute_neighbors(weights: np.ndarray, T: int) -> np.ndarray:
    """Compute T-nearest neighbors for each weight vector.

    Parameters
    ----------
    weights : ndarray, shape (N, M)
    T : int
        Neighborhood size (clamped to N-1 if needed).

    Returns
    -------
    ndarray, shape (N, T)
        Index array of neighbors for each weight vector.
    """
    N = weights.shape[0]
    T = min(T, N - 1)
    # Pairwise Euclidean distances
    dists = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    # For each row, sort and take indices 1..T (skip self at index 0)
    neighbors = np.argsort(dists, axis=1)[:, 1 : T + 1]
    return neighbors


def associate_to_weights(
    F_norm: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Associate each solution to its nearest weight vector by perpendicular distance.

    Parameters
    ----------
    F_norm : ndarray, shape (N, M)
        Normalised objective values.
    weights : ndarray, shape (W, M)
        Weight vectors (unit-simplex).

    Returns
    -------
    ndarray, shape (N,)
        Index of the nearest weight vector for each solution.
    """
    # Perpendicular distance from each point to each reference line (weight direction)
    # d_perp(f, w) = ||f - (f·w / w·w) * w||
    # Since weights are on the simplex they are not unit vectors, so we normalise.
    w_norm = weights / np.linalg.norm(weights, axis=1, keepdims=True)
    # Projection lengths: shape (N, W)
    proj = F_norm @ w_norm.T  # (N, W)
    # Projected points: for each (i, j), proj_point = proj[i,j] * w_norm[j]
    # Distance: ||F_norm[i] - proj_point||
    # Expand for broadcasting
    F_exp = F_norm[:, None, :]  # (N, 1, M)
    w_exp = w_norm[None, :, :]  # (1, W, M)
    proj_exp = proj[:, :, None]  # (N, W, 1)
    diff = F_exp - proj_exp * w_exp  # (N, W, M)
    d_perp = np.linalg.norm(diff, axis=2)  # (N, W)
    return np.argmin(d_perp, axis=1)


def tchebycheff(
    F: np.ndarray, weight: np.ndarray, z_ideal: np.ndarray
) -> np.ndarray | float:
    """Tchebycheff scalarizing function.

    g^tch(x | λ, z*) = max_k { λ_k * |f_k(x) - z*_k| }

    Parameters
    ----------
    F : ndarray, shape (N, M) or (M,)
    weight : ndarray, shape (M,)
    z_ideal : ndarray, shape (M,)

    Returns
    -------
    ndarray, shape (N,) when F is 2-D; numpy scalar when F is 1-D.

    Notes
    -----
    FIX(audit B11): return annotation updated from ``np.ndarray`` to
    ``np.ndarray | float`` to reflect that a 1-D input produces a scalar.
    """
    diff = np.abs(F - z_ideal)
    # Avoid zero weights causing issues
    w = np.maximum(weight, 1e-6)
    return np.max(w * diff, axis=-1)


def auto_partitions(n_pop: int, n_obj: int) -> int:
    """Find the largest H such that C(H+M-1, M-1) <= n_pop.

    This lets us set H from population size automatically.
    """
    from math import comb

    H = 1
    while comb(H + n_obj - 1, n_obj - 1) <= n_pop:
        H += 1
    return H - 1
