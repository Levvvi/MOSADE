"""DE mutation strategies for MOSADE.

Four strategies as defined in the design document:
  S1: DE/current-to-pbest/1  (convergence)
  S2: DE/rand/1              (exploration)
  S3: DE/current-to-rand/1   (rotation-invariant, no crossover)
  S4: DE/rand-to-pbest/2     (convergence + broad sampling)
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class Strategy(IntEnum):
    CURRENT_TO_PBEST_1 = 0
    RAND_1 = 1
    CURRENT_TO_RAND_1 = 2
    RAND_TO_PBEST_2 = 3


NUM_STRATEGIES = len(Strategy)


def mutate_current_to_pbest_1(
    x_i: np.ndarray,
    x_pbest: np.ndarray,
    x_r1: np.ndarray,
    x_r2: np.ndarray,
    F: float,
) -> np.ndarray:
    """v = x_i + F*(x_pbest - x_i) + F*(x_r1 - x_r2)"""
    return x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)


def mutate_rand_1(
    x_r1: np.ndarray,
    x_r2: np.ndarray,
    x_r3: np.ndarray,
    F: float,
) -> np.ndarray:
    """v = x_r1 + F*(x_r2 - x_r3)"""
    return x_r1 + F * (x_r2 - x_r3)


def mutate_current_to_rand_1(
    x_i: np.ndarray,
    x_r1: np.ndarray,
    x_r2: np.ndarray,
    x_r3: np.ndarray,
    F: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """v = x_i + K*(x_r1 - x_i) + F*(x_r2 - x_r3), K ~ U(0,1).

    NOTE: no crossover should be applied after this strategy.
    """
    K = rng.random()
    return x_i + K * (x_r1 - x_i) + F * (x_r2 - x_r3)


def mutate_rand_to_pbest_2(
    x_r1: np.ndarray,
    x_pbest: np.ndarray,
    x_r2: np.ndarray,
    x_r3: np.ndarray,
    F: float,
) -> np.ndarray:
    """v = x_r1 + F*(x_pbest - x_r1) + F*(x_r2 - x_r3)"""
    return x_r1 + F * (x_pbest - x_r1) + F * (x_r2 - x_r3)


def binomial_crossover(
    x: np.ndarray, v: np.ndarray, CR: float, rng: np.random.Generator
) -> np.ndarray:
    """Standard binomial (uniform) crossover."""
    D = x.shape[0]
    j_rand = rng.integers(D)
    mask = rng.random(D) < CR
    mask[j_rand] = True
    u = np.where(mask, v, x)
    return u


def polynomial_mutation(
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    pm: float | None = None,
    eta_m: float = 20.0,
) -> np.ndarray:
    """Polynomial mutation operator.

    Parameters
    ----------
    x : ndarray, shape (D,)
    lower, upper : ndarray, shape (D,)
    rng : Generator
    pm : float, optional
        Per-variable mutation probability.  Default: 1/D.
    eta_m : float
        Distribution index (higher = smaller perturbations).
    """
    D = x.shape[0]
    if pm is None:
        pm = 1.0 / D

    u = x.copy()
    do_mut = rng.random(D) < pm
    for j in np.where(do_mut)[0]:
        # FIX(audit B1): implement the correct NSGA-II polynomial mutation formula.
        # The original code computed delta but never used it; the correct formula
        # requires delta1=(x-lb)/span and delta2=(ub-x)/span to bound perturbations
        # relative to x's position within [lower, upper].
        span = max(upper[j] - lower[j], 1e-30)
        delta1 = (u[j] - lower[j]) / span
        delta2 = (upper[j] - u[j]) / span
        r = rng.random()
        if r < 0.5:
            deltaq = (
                (2.0 * r + (1.0 - 2.0 * r) * (1.0 - delta1) ** (eta_m + 1.0))
                ** (1.0 / (eta_m + 1.0))
            ) - 1.0
        else:
            deltaq = 1.0 - (
                (2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (1.0 - delta2) ** (eta_m + 1.0))
                ** (1.0 / (eta_m + 1.0))
            )
        u[j] = u[j] + deltaq * span
        u[j] = np.clip(u[j], lower[j], upper[j])
    return u


def midpoint_repair(
    u: np.ndarray, x_parent: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Midpoint (bounce-back) bound repair.

    For each violated dimension, set to midpoint of parent and violated bound.
    """
    repaired = u.copy()
    # Replace any NaN/Inf with the parent value before bound checks.
    nan_mask = ~np.isfinite(repaired)
    repaired[nan_mask] = x_parent[nan_mask]
    lo_mask = repaired < lower
    hi_mask = repaired > upper
    repaired[lo_mask] = (x_parent[lo_mask] + lower[lo_mask]) / 2.0
    repaired[hi_mask] = (x_parent[hi_mask] + upper[hi_mask]) / 2.0
    return repaired
