"""DASCMOP benchmark suite (Fan et al., 2020).

Difficulty-Adjustable and Scalable Constrained Multi-objective Problems.
9 problems (DASCMOP1-9) with adjustable difficulty via parameter setting.

Difficulty triplet (eta, zeta, gamma) controls:
  - eta:   feasibility ratio (larger eta = fewer feasible solutions)
  - zeta:  convergence hardness (controls distance ring around PF)
  - gamma: diversity hardness (controls exclusion zones in objective space)

Standard difficulty settings map to difficulty levels 1-16.

DASCMOP1-6 are bi-objective with 11 constraints.
DASCMOP7-9 are tri-objective with 7 constraints.

Reference: Fan, Z., Li, W., Cai, X., et al. (2020). "Difficulty Adjustable
and Scalable Constrained Multiobjective Test Problem Toolkit."
Evolutionary Computation, 28(3), 339-378.
"""

from __future__ import annotations

import numpy as np

from mosade.problems.base import Problem


# ======================================================================
# Difficulty settings (matching pymoo reference implementation)
# ======================================================================

# 16 standard difficulty triplets (eta, zeta, gamma).
# Levels 1-4: mild, 5-8: moderate, 9-12: hard, 13-16: extreme/mixed.
DIFFICULTY_SETTINGS = {
    1:  (0.25, 0.0,  0.0),
    2:  (0.0,  0.25, 0.0),
    3:  (0.0,  0.0,  0.25),
    4:  (0.25, 0.25, 0.25),
    5:  (0.5,  0.0,  0.0),
    6:  (0.0,  0.5,  0.0),
    7:  (0.0,  0.0,  0.5),
    8:  (0.5,  0.5,  0.5),
    9:  (0.75, 0.0,  0.0),
    10: (0.0,  0.75, 0.0),
    11: (0.0,  0.0,  0.75),
    12: (0.75, 0.75, 0.75),
    13: (0.0,  1.0,  0.0),
    14: (0.5,  1.0,  0.0),
    15: (0.0,  1.0,  0.5),
    16: (0.5,  1.0,  0.5),
}


def _get_difficulty(level: int) -> tuple[float, float, float]:
    """Look up (eta, zeta, gamma) triplet for a given difficulty level."""
    if level not in DIFFICULTY_SETTINGS:
        raise ValueError(f"Difficulty level must be 1-16, got {level}")
    return DIFFICULTY_SETTINGS[level]


# ======================================================================
# Distance functions
# ======================================================================

def _g1(X: np.ndarray, n_obj: int) -> np.ndarray:
    """Simple distance function (used by DASCMOP1-3).

    Parameters
    ----------
    X : ndarray, shape (N, n_var)
    n_obj : int

    Returns
    -------
    g : ndarray, shape (N,)
    """
    contrib = (X[:, n_obj - 1:] - np.sin(0.5 * np.pi * X[:, 0:1])) ** 2
    return contrib.sum(axis=1)


def _g2(X: np.ndarray, n_obj: int, n_var: int) -> np.ndarray:
    """Rastrigin-like distance function (used by DASCMOP4-8).

    Parameters
    ----------
    X : ndarray, shape (N, n_var)
    n_obj : int
    n_var : int

    Returns
    -------
    g : ndarray, shape (N,)
    """
    z = X[:, n_obj - 1:] - 0.5
    contrib = z ** 2 - np.cos(20.0 * np.pi * z)
    return (n_var - n_obj + 1) + contrib.sum(axis=1)


def _g3(X: np.ndarray, n_obj: int, n_var: int) -> np.ndarray:
    """Correlated distance function (used by DASCMOP9).

    Parameters
    ----------
    X : ndarray, shape (N, n_var)
    n_obj : int
    n_var : int

    Returns
    -------
    g : ndarray, shape (N,)
    """
    j = np.arange(n_obj - 1, n_var) + 1
    contrib = (
        X[:, n_obj - 1:]
        - np.cos(0.25 * j / n_var * np.pi * (X[:, 0:1] + X[:, 1:2]))
    ) ** 2
    return contrib.sum(axis=1)


# ======================================================================
# Constraint functions
# ======================================================================

def _constraints_biobj(
    X: np.ndarray, f0: np.ndarray, f1: np.ndarray, g: np.ndarray,
    eta: float, zeta: float, gamma: float,
) -> np.ndarray:
    """Build 11 constraints for DASCMOP1-6 (bi-objective).

    Convention: G_j <= 0 means feasible.

    Layout:
      0:    sinusoidal feasibility band on x_0 (controlled by eta)
      1:    distance ring around PF (controlled by zeta)
      2-10: nine elliptical exclusion zones (controlled by gamma)
    """
    N = X.shape[0]
    G = np.zeros((N, 11))

    # --- Constraint 0: sinusoidal feasibility on x_0 ---
    b = 2.0 * eta - 1.0
    G[:, 0] = b - np.sin(20.0 * np.pi * X[:, 0])

    # --- Constraint 1: distance ring ---
    d = 0.5 if zeta != 0 else 0.0
    if zeta > 0:
        e = d - np.log(zeta)
    else:
        e = 1e30

    if zeta == 1.0:
        G[:, 1] = np.abs(e - g) - 1e-4
    else:
        G[:, 1] = -((e - g) * (g - d))

    # --- Constraints 2-10: nine elliptical exclusion zones ---
    r = 0.5 * gamma
    p_k = np.array([0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0])
    q_k = np.array([1.5, 0.5, 2.5, 1.5, 0.5, 3.5, 2.5, 1.5, 0.5])
    a_k2 = 0.3
    b_k2 = 1.2
    theta = -0.25 * np.pi

    f0_2d = f0.reshape(-1, 1)  # (N, 1)
    f1_2d = f1.reshape(-1, 1)  # (N, 1)

    dx = f0_2d - p_k  # (N, 9)
    dy = f1_2d - q_k  # (N, 9)

    x_rot = dx * np.cos(theta) - dy * np.sin(theta)
    y_rot = dx * np.sin(theta) + dy * np.cos(theta)

    expr = x_rot ** 2 / a_k2 + y_rot ** 2 / b_k2
    G[:, 2:] = r - expr

    return G


def _constraints_triobj(
    X: np.ndarray, f0: np.ndarray, f1: np.ndarray, f2: np.ndarray,
    g: np.ndarray, eta: float, zeta: float, gamma: float,
) -> np.ndarray:
    """Build 7 constraints for DASCMOP7-9 (tri-objective).

    Convention: G_j <= 0 means feasible.

    Layout:
      0: sinusoidal on x_0 (eta)
      1: sinusoidal on x_1 (eta)
      2: distance ring (zeta)
      3-6: four spherical exclusion zones (gamma)
    """
    N = X.shape[0]
    G = np.zeros((N, 7))

    b = 2.0 * eta - 1.0
    G[:, 0] = b - np.sin(20.0 * np.pi * X[:, 0])
    G[:, 1] = b - np.cos(20.0 * np.pi * X[:, 1])

    d = 0.5 if zeta != 0 else 0.0
    if zeta > 0:
        e = d - np.log(zeta)
    else:
        e = 1e30

    if zeta == 1.0:
        G[:, 2] = np.abs(e - g) - 1e-4
    else:
        G[:, 2] = -((e - g) * (g - d))

    r = 0.5 * gamma
    inv3 = 1.0 / np.sqrt(3.0)
    x_k = np.array([1.0, 0.0, 0.0, inv3])
    y_k = np.array([0.0, 1.0, 0.0, inv3])
    z_k = np.array([0.0, 0.0, 1.0, inv3])

    f0_2d = f0.reshape(-1, 1)
    f1_2d = f1.reshape(-1, 1)
    f2_2d = f2.reshape(-1, 1)

    expr = (f0_2d - x_k) ** 2 + (f1_2d - y_k) ** 2 + (f2_2d - z_k) ** 2
    G[:, 3:] = r ** 2 - expr

    return G


# ======================================================================
# DASCMOP base
# ======================================================================

class _DASCMOPBase(Problem):
    """Base class for DASCMOP problems."""

    def __init__(self, n_var: int, n_obj: int, n_constr: int,
                 difficulty: int) -> None:
        self._difficulty = difficulty
        self._eta, self._zeta, self._gamma = _get_difficulty(difficulty)
        super().__init__(
            n_var=n_var, n_obj=n_obj, n_constr=n_constr,
            lower=np.zeros(n_var), upper=np.ones(n_var),
        )


# ======================================================================
# DASCMOP1-6: bi-objective, 11 constraints
# ======================================================================

class DASCMOP1(_DASCMOPBase):
    """DASCMOP1: Convex PF (1 - x^2) with g1 distance, 11 constraints."""

    def __init__(self, n_var: int = 30, difficulty: int = 8) -> None:
        super().__init__(n_var=n_var, n_obj=2, n_constr=11, difficulty=difficulty)

    def _evaluate(self, X: np.ndarray):
        g = _g1(X, self.n_obj)
        f0 = X[:, 0] + g
        f1 = 1.0 - X[:, 0] ** 2 + g
        F = np.column_stack([f0, f1])
        G = _constraints_biobj(X, f0, f1, g, self._eta, self._zeta, self._gamma)
        return F, G


class DASCMOP2(_DASCMOPBase):
    """DASCMOP2: Concave PF (1 - sqrt(x)) with g1 distance, 11 constraints."""

    def __init__(self, n_var: int = 30, difficulty: int = 8) -> None:
        super().__init__(n_var=n_var, n_obj=2, n_constr=11, difficulty=difficulty)

    def _evaluate(self, X: np.ndarray):
        g = _g1(X, self.n_obj)
        f0 = X[:, 0] + g
        f1 = 1.0 - np.sqrt(X[:, 0]) + g
        F = np.column_stack([f0, f1])
        G = _constraints_biobj(X, f0, f1, g, self._eta, self._zeta, self._gamma)
        return F, G


class DASCMOP3(_DASCMOPBase):
    """DASCMOP3: Disconnected PF with g1 distance, 11 constraints."""

    def __init__(self, n_var: int = 30, difficulty: int = 8) -> None:
        super().__init__(n_var=n_var, n_obj=2, n_constr=11, difficulty=difficulty)

    def _evaluate(self, X: np.ndarray):
        g = _g1(X, self.n_obj)
        f0 = X[:, 0] + g
        f1 = 1.0 - np.sqrt(X[:, 0]) + 0.5 * np.abs(np.sin(5.0 * np.pi * X[:, 0])) + g
        F = np.column_stack([f0, f1])
        G = _constraints_biobj(X, f0, f1, g, self._eta, self._zeta, self._gamma)
        return F, G


class DASCMOP4(_DASCMOPBase):
    """DASCMOP4: Convex PF (1 - x^2) with g2 (Rastrigin) distance, 11 constraints."""

    def __init__(self, n_var: int = 30, difficulty: int = 8) -> None:
        super().__init__(n_var=n_var, n_obj=2, n_constr=11, difficulty=difficulty)

    def _evaluate(self, X: np.ndarray):
        g = _g2(X, self.n_obj, self.n_var)
        f0 = X[:, 0] + g
        f1 = 1.0 - X[:, 0] ** 2 + g
        F = np.column_stack([f0, f1])
        G = _constraints_biobj(X, f0, f1, g, self._eta, self._zeta, self._gamma)
        return F, G


class DASCMOP5(_DASCMOPBase):
    """DASCMOP5: Concave PF (1 - sqrt(x)) with g2 (Rastrigin) distance, 11 constraints."""

    def __init__(self, n_var: int = 30, difficulty: int = 8) -> None:
        super().__init__(n_var=n_var, n_obj=2, n_constr=11, difficulty=difficulty)

    def _evaluate(self, X: np.ndarray):
        g = _g2(X, self.n_obj, self.n_var)
        f0 = X[:, 0] + g
        f1 = 1.0 - np.sqrt(X[:, 0]) + g
        F = np.column_stack([f0, f1])
        G = _constraints_biobj(X, f0, f1, g, self._eta, self._zeta, self._gamma)
        return F, G


class DASCMOP6(_DASCMOPBase):
    """DASCMOP6: Disconnected PF with g2 (Rastrigin) distance, 11 constraints."""

    def __init__(self, n_var: int = 30, difficulty: int = 8) -> None:
        super().__init__(n_var=n_var, n_obj=2, n_constr=11, difficulty=difficulty)

    def _evaluate(self, X: np.ndarray):
        g = _g2(X, self.n_obj, self.n_var)
        f0 = X[:, 0] + g
        f1 = 1.0 - np.sqrt(X[:, 0]) + 0.5 * np.abs(np.sin(5.0 * np.pi * X[:, 0])) + g
        F = np.column_stack([f0, f1])
        G = _constraints_biobj(X, f0, f1, g, self._eta, self._zeta, self._gamma)
        return F, G


# ======================================================================
# DASCMOP7-9: tri-objective, 7 constraints
# ======================================================================

class DASCMOP7(_DASCMOPBase):
    """DASCMOP7: Linear tri-objective PF with g2 distance, 7 constraints."""

    def __init__(self, n_var: int = 30, difficulty: int = 8) -> None:
        super().__init__(n_var=n_var, n_obj=3, n_constr=7, difficulty=difficulty)

    def _evaluate(self, X: np.ndarray):
        g = _g2(X, self.n_obj, self.n_var)
        f0 = X[:, 0] * X[:, 1] + g
        f1 = X[:, 1] * (1.0 - X[:, 0]) + g
        f2 = 1.0 - X[:, 1] + g
        F = np.column_stack([f0, f1, f2])
        G = _constraints_triobj(X, f0, f1, f2, g, self._eta, self._zeta, self._gamma)
        return F, G


class DASCMOP8(_DASCMOPBase):
    """DASCMOP8: Spherical tri-objective PF with g2 distance, 7 constraints."""

    def __init__(self, n_var: int = 30, difficulty: int = 8) -> None:
        super().__init__(n_var=n_var, n_obj=3, n_constr=7, difficulty=difficulty)

    def _evaluate(self, X: np.ndarray):
        g = _g2(X, self.n_obj, self.n_var)
        f0 = np.cos(0.5 * np.pi * X[:, 0]) * np.cos(0.5 * np.pi * X[:, 1]) + g
        f1 = np.cos(0.5 * np.pi * X[:, 0]) * np.sin(0.5 * np.pi * X[:, 1]) + g
        f2 = np.sin(0.5 * np.pi * X[:, 0]) + g
        F = np.column_stack([f0, f1, f2])
        G = _constraints_triobj(X, f0, f1, f2, g, self._eta, self._zeta, self._gamma)
        return F, G


class DASCMOP9(_DASCMOPBase):
    """DASCMOP9: Spherical tri-objective PF with g3 distance, 7 constraints."""

    def __init__(self, n_var: int = 30, difficulty: int = 8) -> None:
        super().__init__(n_var=n_var, n_obj=3, n_constr=7, difficulty=difficulty)

    def _evaluate(self, X: np.ndarray):
        g = _g3(X, self.n_obj, self.n_var)
        f0 = np.cos(0.5 * np.pi * X[:, 0]) * np.cos(0.5 * np.pi * X[:, 1]) + g
        f1 = np.cos(0.5 * np.pi * X[:, 0]) * np.sin(0.5 * np.pi * X[:, 1]) + g
        f2 = np.sin(0.5 * np.pi * X[:, 0]) + g
        F = np.column_stack([f0, f1, f2])
        G = _constraints_triobj(X, f0, f1, f2, g, self._eta, self._zeta, self._gamma)
        return F, G
