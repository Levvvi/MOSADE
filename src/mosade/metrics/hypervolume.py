"""Hypervolume indicator.

For bi-objective problems, an exact O(N log N) sweep-line algorithm is used.
For 3-objective problems, an exact slicing algorithm is used (O(N^2 log N)).
For 4+ objectives, a Monte Carlo estimator is used (approximate — see warning
in `_hv_mc`).
"""

from __future__ import annotations

import numpy as np


def hypervolume(F: np.ndarray, ref: np.ndarray, seed: int | None = None) -> float:
    """Compute the hypervolume indicator.

    Parameters
    ----------
    F : ndarray, shape (N, M)
        Objective vectors of the approximation set (minimisation).
    ref : ndarray, shape (M,)
        Reference point (must dominate all points in F for a meaningful result).
    seed : int or None, optional
        RNG seed for the Monte Carlo estimator (4+ objectives only).
        Ignored for M <= 3 where exact algorithms are used.

    Returns
    -------
    float
        Hypervolume value.  Returns 0.0 if F is empty or no point
        is dominated by the reference point.
    """
    if F.shape[0] == 0:
        return 0.0

    mask = np.all(F <= ref, axis=1)
    F = F[mask]
    if F.shape[0] == 0:
        return 0.0

    M = F.shape[1]
    if M == 2:
        return _hv_2d(F, ref)
    if M == 3:
        return _hv_3d(F, ref)
    return _hv_mc(F, ref, seed=seed)


def _hv_2d(F: np.ndarray, ref: np.ndarray) -> float:
    """Exact 2-D hypervolume via sweep line.  O(N log N)."""
    order = np.argsort(F[:, 0])
    F_sorted = F[order]

    hv = 0.0
    prev_f2 = ref[1]
    for i in range(F_sorted.shape[0]):
        if F_sorted[i, 1] < prev_f2:
            hv += (ref[0] - F_sorted[i, 0]) * (prev_f2 - F_sorted[i, 1])
            prev_f2 = F_sorted[i, 1]
    return float(hv)


def _hv_3d(F: np.ndarray, ref: np.ndarray) -> float:
    """Exact 3-D hypervolume via slicing.  O(N^2 log N).

    Sorts points by the third objective, then sweeps from smallest to largest
    f3 value.  Each consecutive pair of f3 levels defines a slab whose volume
    equals the 2-D hypervolume of the (f1, f2) projection of all points seen
    so far, multiplied by the slab height.
    """
    order = np.argsort(F[:, 2])
    F_sorted = F[order]
    N = F_sorted.shape[0]
    ref_2d = ref[:2]

    hv = 0.0
    for i in range(N):
        next_f3 = F_sorted[i + 1, 2] if i + 1 < N else ref[2]
        h = next_f3 - F_sorted[i, 2]
        if h <= 0.0:
            continue
        hv += h * _hv_2d(F_sorted[: i + 1, :2], ref_2d)

    return float(hv)


def _hv_mc(F: np.ndarray, ref: np.ndarray, seed: int | None = None) -> float:
    """Monte Carlo hypervolume estimator for M >= 4 objectives.

    .. warning::
        This is an **approximate** method.  Results carry statistical error
        (standard deviation ~ volume_box / sqrt(n_samples)).  For
        high-accuracy results on 4+ objectives, replace this with an
        exact algorithm such as WFG or integrate a library like pygmo.

    Parameters
    ----------
    F : ndarray, shape (N, M)
    ref : ndarray, shape (M,)
    seed : int or None
        Seed for the internal RNG.  Pass a fixed integer for reproducible
        estimates; leave as None for a fresh random seed each call.

    Returns
    -------
    float
        Estimated hypervolume.
    """
    N, M = F.shape
    n_samples = 500_000
    rng = np.random.default_rng(seed)

    ideal = F.min(axis=0)
    volume_box = float(np.prod(ref - ideal))
    if volume_box <= 0:
        return 0.0

    samples = rng.uniform(ideal, ref, size=(n_samples, M))
    dominated = np.zeros(n_samples, dtype=bool)
    for i in range(N):
        dominated |= np.all(samples >= F[i], axis=1)
    return volume_box * float(dominated.mean())
