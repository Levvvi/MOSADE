"""DTLZ benchmark suite (Deb, Thiele, Laumanns, Zitzler, 2005).

Scalable to any number of objectives.  Default: 3 objectives.
"""

from __future__ import annotations

import numpy as np

from mosade.problems.base import Problem


class DTLZ1(Problem):
    """Linear Pareto front with many local fronts."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3) -> None:
        if n_var is None:
            n_var = n_obj + 4  # standard k=5
        super().__init__(n_var=n_var, n_obj=n_obj)
        self._k = n_var - n_obj + 1

    def _evaluate(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        N = X.shape[0]
        M = self.n_obj
        k = self._k
        xm = X[:, M - 1:]  # last k variables

        g = 100.0 * (
            k + np.sum((xm - 0.5) ** 2 - np.cos(20.0 * np.pi * (xm - 0.5)), axis=1)
        )

        F = np.zeros((N, M))
        for i in range(M):
            f = 0.5 * (1.0 + g)
            for j in range(M - 1 - i):
                f = f * X[:, j]
            if i > 0:
                f = f * (1.0 - X[:, M - 1 - i])
            F[:, i] = f

        return F, None

    def pareto_front(self, n_points: int = 500):
        """Linear hyperplane: sum(f_i) = 0.5, f_i >= 0."""
        M = self.n_obj
        if M == 2:
            f1 = np.linspace(0, 0.5, n_points)
            f2 = 0.5 - f1
            return np.column_stack([f1, f2])
        # General M: Das-Dennis on the simplex, scaled to 0.5
        from mosade.algorithm.decomposition import das_dennis, auto_partitions

        H = auto_partitions(n_points, M)
        H = max(H, 1)
        W = das_dennis(H, M)
        return W * 0.5


class DTLZ2(Problem):
    """Spherical (concave) Pareto front."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3) -> None:
        if n_var is None:
            n_var = n_obj + 9  # standard k=10
        super().__init__(n_var=n_var, n_obj=n_obj)

    def _evaluate(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        N = X.shape[0]
        M = self.n_obj
        xm = X[:, M - 1:]

        g = np.sum((xm - 0.5) ** 2, axis=1)

        F = np.zeros((N, M))
        for i in range(M):
            f = 1.0 + g
            for j in range(M - 1 - i):
                f = f * np.cos(X[:, j] * np.pi / 2.0)
            if i > 0:
                f = f * np.sin(X[:, M - 1 - i] * np.pi / 2.0)
            F[:, i] = f

        return F, None

    def pareto_front(self, n_points: int = 500):
        if self.n_obj == 2:
            theta = np.linspace(0, np.pi / 2, n_points)
            return np.column_stack([np.cos(theta), np.sin(theta)])
        return None


class DTLZ3(Problem):
    """Spherical Pareto front with many local fronts (same g as DTLZ1)."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3) -> None:
        if n_var is None:
            n_var = n_obj + 9  # standard k=10
        super().__init__(n_var=n_var, n_obj=n_obj)
        self._k = n_var - n_obj + 1

    def _evaluate(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        N = X.shape[0]
        M = self.n_obj
        k = self._k
        xm = X[:, M - 1:]

        g = 100.0 * (
            k + np.sum((xm - 0.5) ** 2 - np.cos(20.0 * np.pi * (xm - 0.5)), axis=1)
        )

        F = np.zeros((N, M))
        for i in range(M):
            f = 1.0 + g
            for j in range(M - 1 - i):
                f = f * np.cos(X[:, j] * np.pi / 2.0)
            if i > 0:
                f = f * np.sin(X[:, M - 1 - i] * np.pi / 2.0)
            F[:, i] = f

        return F, None


class DTLZ4(Problem):
    """Spherical Pareto front with biased density.

    Parameters
    ----------
    alpha : float
        Density bias exponent.  Higher values concentrate solutions near
        the f_M axis. Default 100 (standard value from the DTLZ reference).
    """

    def __init__(self, n_var: int | None = None, n_obj: int = 3,
                 alpha: float = 100.0) -> None:
        if n_var is None:
            n_var = n_obj + 9  # standard k=10
        super().__init__(n_var=n_var, n_obj=n_obj)
        self._alpha = alpha

    def _evaluate(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        N = X.shape[0]
        M = self.n_obj
        xm = X[:, M - 1:]

        g = np.sum((xm - 0.5) ** 2, axis=1)

        Xb = X[:, :M - 1] ** self._alpha

        F = np.zeros((N, M))
        for i in range(M):
            f = 1.0 + g
            for j in range(M - 1 - i):
                f = f * np.cos(Xb[:, j] * np.pi / 2.0)
            if i > 0:
                f = f * np.sin(Xb[:, M - 1 - i] * np.pi / 2.0)
            F[:, i] = f

        return F, None
