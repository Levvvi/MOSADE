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


class DTLZ7(Problem):
    """Disconnected Pareto front benchmark.

    The standard DTLZ7 formulation has ``k = 20`` distance variables by
    default.  The Pareto-optimal set is obtained at ``x_M, ..., x_D = 0`` and
    produces a disconnected objective-space front.
    """

    def __init__(self, n_var: int | None = None, n_obj: int = 3) -> None:
        if n_var is None:
            n_var = n_obj + 19  # standard k=20
        super().__init__(n_var=n_var, n_obj=n_obj)
        self._k = n_var - n_obj + 1

    def _evaluate(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        N = X.shape[0]
        M = self.n_obj
        xm = X[:, M - 1:]

        g = 1.0 + 9.0 * np.sum(xm, axis=1) / self._k
        F = np.zeros((N, M))
        F[:, : M - 1] = X[:, : M - 1]

        h_terms = (
            (F[:, : M - 1] / (1.0 + g[:, None]))
            * (1.0 + np.sin(3.0 * np.pi * F[:, : M - 1]))
        )
        h = M - np.sum(h_terms, axis=1)
        F[:, M - 1] = (1.0 + g) * h
        return F, None

    def pareto_front(self, n_points: int = 500) -> np.ndarray | None:
        """Return a deterministic approximation of the disconnected PF.

        For the default three-objective case a dense grid is filtered by
        non-domination.  For higher objective counts, deterministic uniform
        samples are used to avoid an exponential grid.
        """
        from mosade.algorithm.selection import nondominated_mask

        M = self.n_obj
        if M < 2:
            return None

        intervals = ((0.0, 0.251412), (0.631627, 0.859401))
        n_regions = 2 ** (M - 1)

        if M <= 4:
            import itertools

            side = max(8, int(np.ceil((max(n_points, 1) / n_regions) ** (1.0 / (M - 1)))))
            blocks = []
            for region in itertools.product(intervals, repeat=M - 1):
                axes = [np.linspace(lo, hi, side) for lo, hi in region]
                mesh = np.meshgrid(*axes, indexing="ij")
                blocks.append(np.column_stack([item.ravel() for item in mesh]))
            prefix = np.vstack(blocks)
        else:
            rng = np.random.default_rng(707)
            n_candidates = max(50_000, n_points * 200)
            choices = rng.integers(0, 2, size=(n_candidates, M - 1))
            lows = np.take([intervals[0][0], intervals[1][0]], choices)
            highs = np.take([intervals[0][1], intervals[1][1]], choices)
            prefix = rng.uniform(lows, highs)

        g = 1.0
        h_terms = (prefix / (1.0 + g)) * (1.0 + np.sin(3.0 * np.pi * prefix))
        f_last = (1.0 + g) * (M - np.sum(h_terms, axis=1))
        F = np.column_stack([prefix, f_last])
        F = F[np.isfinite(F).all(axis=1) & (F[:, -1] >= 0.0)]
        if F.size == 0:
            return None

        F = F[nondominated_mask(F)]
        if F.shape[0] <= n_points:
            return F

        order = np.lexsort(tuple(F[:, idx] for idx in range(M - 1, -1, -1)))
        idx = np.linspace(0, len(order) - 1, n_points, dtype=int)
        return F[order[idx]]
