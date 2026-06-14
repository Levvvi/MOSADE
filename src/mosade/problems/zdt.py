"""ZDT benchmark suite (Zitzler, Deb, Thiele, 2000).

All problems are bi-objective, unconstrained, with 30 decision variables
by default. Bounds are [0,1]^n except ZDT4 where x_2..x_n ∈ [-5, 5].
"""

from __future__ import annotations

import numpy as np

from mosade.problems.base import Problem


class ZDT1(Problem):
    """Convex Pareto front."""

    def __init__(self, n_var: int = 30) -> None:
        super().__init__(n_var=n_var, n_obj=2)

    def _evaluate(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        f1 = X[:, 0]
        g = 1.0 + 9.0 * np.mean(X[:, 1:], axis=1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.column_stack([f1, f2]), None

    def pareto_front(self, n_points: int = 500):
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])


class ZDT2(Problem):
    """Non-convex (concave) Pareto front."""

    def __init__(self, n_var: int = 30) -> None:
        super().__init__(n_var=n_var, n_obj=2)

    def _evaluate(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        f1 = X[:, 0]
        g = 1.0 + 9.0 * np.mean(X[:, 1:], axis=1)
        f2 = g * (1.0 - (f1 / g) ** 2)
        return np.column_stack([f1, f2]), None

    def pareto_front(self, n_points: int = 500):
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - f1**2
        return np.column_stack([f1, f2])


class ZDT3(Problem):
    """Disconnected Pareto front."""

    def __init__(self, n_var: int = 30) -> None:
        super().__init__(n_var=n_var, n_obj=2)

    def _evaluate(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        f1 = X[:, 0]
        g = 1.0 + 9.0 * np.mean(X[:, 1:], axis=1)
        f2 = g * (1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1))
        return np.column_stack([f1, f2]), None

    def pareto_front(self, n_points: int = 500):
        # The PF of ZDT3 is disconnected: five segments on the f1 axis.
        regions = [
            (0.0, 0.0830015349),
            (0.1822287280, 0.2577623634),
            (0.4093136748, 0.4538821041),
            (0.6183967944, 0.6525117038),
            (0.8233317983, 0.8518328654),
        ]
        pts_per = n_points // len(regions)
        segments = []
        for lo, hi in regions:
            f1 = np.linspace(lo, hi, max(pts_per, 10))
            f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)
            segments.append(np.column_stack([f1, f2]))
        return np.vstack(segments)


class ZDT4(Problem):
    """Multi-modal (many local Pareto fronts)."""

    def __init__(self, n_var: int = 10) -> None:
        lower = np.full(n_var, -5.0)
        lower[0] = 0.0
        upper = np.full(n_var, 5.0)
        upper[0] = 1.0
        super().__init__(n_var=n_var, n_obj=2, lower=lower, upper=upper)

    def _evaluate(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        f1 = X[:, 0]
        g = 1.0 + 10.0 * (self.n_var - 1) + np.sum(
            X[:, 1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * X[:, 1:]), axis=1
        )
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.column_stack([f1, f2]), None

    def pareto_front(self, n_points: int = 500):
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])


class ZDT6(Problem):
    """Non-uniform density on the Pareto front; biased search space."""

    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var=n_var, n_obj=2)

    def _evaluate(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        f1 = 1.0 - np.exp(-4.0 * X[:, 0]) * np.sin(6.0 * np.pi * X[:, 0]) ** 6
        g = 1.0 + 9.0 * (np.mean(X[:, 1:], axis=1) ** 0.25)
        f2 = g * (1.0 - (f1 / g) ** 2)
        return np.column_stack([f1, f2]), None

    def pareto_front(self, n_points: int = 500):
        f1 = np.linspace(0.280775, 1.0, n_points)
        f2 = 1.0 - f1**2
        return np.column_stack([f1, f2])
