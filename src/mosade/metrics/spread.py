"""Spread / spacing metric for diversity assessment.

Measures how uniformly the approximation set covers the Pareto front.
"""

from __future__ import annotations

import numpy as np


def spread(F: np.ndarray, PF: np.ndarray | None = None) -> float:
    """Compute a diversity indicator for an approximation set.

    For bi-objective problems (M=2): computes the Spread (Delta) indicator
    from Deb et al. (2002), measuring extent and uniformity of coverage.

    For M > 2: falls back to the Schott Spacing metric, which measures
    variance of minimum-neighbour distances.  The two metrics have different
    semantics and scales; callers should not compare spread values across
    problems with different numbers of objectives.

    Parameters
    ----------
    F : ndarray, shape (N, M)
        Approximation set.
    PF : ndarray, shape (P, M), optional
        True Pareto front.  Used only for the M=2 path to compute distances
        from the approximation set's extremes to the true PF extremes.

    Returns
    -------
    float
        Lower is better in both cases.

    Notes
    -----
    FIX(audit B12): the original docstring described only the M=2 Delta
    indicator without mentioning the M>2 fallback to Spacing, which has
    different semantics.  Both cases are now documented explicitly.
    """
    if F.shape[0] <= 1:
        return float("inf")

    if F.shape[1] != 2:
        # For M>2, fall back to Schott Spacing metric (different semantics — see docstring).
        return spacing(F)

    # Sort by first objective
    order = np.argsort(F[:, 0])
    F_sorted = F[order]

    # Consecutive distances
    dists = np.linalg.norm(np.diff(F_sorted, axis=0), axis=1)
    d_mean = dists.mean()

    if d_mean < 1e-30:
        return 0.0

    # Extreme point distances
    if PF is not None and PF.shape[0] > 0:
        pf_sorted = PF[np.argsort(PF[:, 0])]
        d_first = np.linalg.norm(F_sorted[0] - pf_sorted[0])
        d_last = np.linalg.norm(F_sorted[-1] - pf_sorted[-1])
    else:
        d_first = 0.0
        d_last = 0.0

    numerator = d_first + d_last + np.sum(np.abs(dists - d_mean))
    denominator = d_first + d_last + len(dists) * d_mean

    if denominator < 1e-30:
        return 0.0
    return float(numerator / denominator)


def spacing(F: np.ndarray) -> float:
    """Schott's Spacing metric (general M).

    Measures variance of minimum neighbour distances.
    Lower is better (more uniform).
    """
    if F.shape[0] <= 1:
        return float("inf")

    N = F.shape[0]
    dists = np.full(N, np.inf)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = np.linalg.norm(F[i] - F[j])
            if d < dists[i]:
                dists[i] = d

    return float(np.std(dists))
