"""Generational Distance (GD).

GD measures how far the approximation set is from the true Pareto front.
Unlike IGD (which measures coverage of the front), GD rewards closeness of
the approximation set's own members to the front and is not Pareto-compliant.

Reference: Van Veldhuizen & Lamont (1998).
"""

from __future__ import annotations

import numpy as np


def gd(F: np.ndarray, PF: np.ndarray) -> float:
    """Compute the Generational Distance.

    .. math::

        GD(F, PF) = \\frac{\\sqrt{\\sum_{f \\in F} d(f, PF)^2}}{|F|}

    where :math:`d(f, PF)` is the minimum Euclidean distance from *f* to
    any point on *PF*.

    Parameters
    ----------
    F : ndarray, shape (N, M)
        Approximation set (the algorithm's output).
    PF : ndarray, shape (P, M)
        Reference Pareto front.

    Returns
    -------
    float
        GD value.  Lower is better.  Returns ``inf`` if *F* is empty.
    """
    if F.shape[0] == 0:
        return float("inf")

    # For each point in F, find its minimum distance to any PF point.
    # Broadcasting: (N, 1, M) - (1, P, M) -> (N, P, M)
    diff = F[:, None, :] - PF[None, :, :]
    dists = np.linalg.norm(diff, axis=2)  # (N, P)
    min_dists = dists.min(axis=1)         # (N,)
    return float(np.sqrt(np.sum(min_dists ** 2)) / F.shape[0])
