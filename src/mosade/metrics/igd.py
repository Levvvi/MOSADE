"""Inverted Generational Distance (IGD and IGD+).

IGD measures how well the approximation set covers the true Pareto front.
IGD+ is the Pareto-compliant variant (Ishibuchi et al., 2015).
"""

from __future__ import annotations

import numpy as np


def igd(F: np.ndarray, PF: np.ndarray) -> float:
    """Compute the Inverted Generational Distance.

    Parameters
    ----------
    F : ndarray, shape (N, M)
        Approximation set.
    PF : ndarray, shape (P, M)
        Reference Pareto front.

    Returns
    -------
    float
        Mean minimum Euclidean distance from PF points to F.
        Lower is better.
    """
    if F.shape[0] == 0:
        return float("inf")

    # For each PF point, find the minimum distance to any F point
    # Using broadcasting: (P, 1, M) - (1, N, M) -> (P, N, M)
    diff = PF[:, None, :] - F[None, :, :]
    dists = np.linalg.norm(diff, axis=2)  # (P, N)
    min_dists = dists.min(axis=1)  # (P,)
    return float(min_dists.mean())


def igd_plus(F: np.ndarray, PF: np.ndarray) -> float:
    """Compute IGD+ (Pareto-compliant variant).

    Uses max(f_k - pf_k, 0) instead of |f_k - pf_k| so that
    solutions dominating a PF point get zero contribution from that point.

    Parameters
    ----------
    F : ndarray, shape (N, M)
    PF : ndarray, shape (P, M)

    Returns
    -------
    float
        Lower is better.
    """
    if F.shape[0] == 0:
        return float("inf")

    diff = F[None, :, :] - PF[:, None, :]  # (P, N, M)
    diff_plus = np.maximum(diff, 0.0)
    dists = np.linalg.norm(diff_plus, axis=2)  # (P, N)
    min_dists = dists.min(axis=1)
    return float(min_dists.mean())
