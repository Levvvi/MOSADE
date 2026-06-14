"""Base class for multi-objective optimization problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProblemSpec:
    """Immutable specification of a problem's search space and objectives."""

    n_var: int
    n_obj: int
    n_constr: int
    lower: np.ndarray  # shape (n_var,)
    upper: np.ndarray  # shape (n_var,)

    def __post_init__(self) -> None:
        # FIX(audit B10): use ValueError instead of assert so validation is never
        # silently disabled by python -O.
        if self.lower.shape != (self.n_var,):
            raise ValueError(
                f"lower.shape {self.lower.shape} != (n_var={self.n_var},)"
            )
        if self.upper.shape != (self.n_var,):
            raise ValueError(
                f"upper.shape {self.upper.shape} != (n_var={self.n_var},)"
            )
        if not np.all(self.lower <= self.upper):
            raise ValueError("lower must be <= upper for all dimensions")


class Problem(ABC):
    """Abstract base class for multi-objective optimization problems.

    Subclasses must implement:
        _evaluate(x) -> (objectives, constraints)

    Optionally override:
        pareto_front(n_points) -> np.ndarray  for known analytical PFs.
    """

    def __init__(self, n_var: int, n_obj: int, n_constr: int = 0,
                 lower: np.ndarray | None = None,
                 upper: np.ndarray | None = None) -> None:
        if lower is None:
            lower = np.zeros(n_var)
        if upper is None:
            upper = np.ones(n_var)
        self.spec = ProblemSpec(
            n_var=n_var, n_obj=n_obj, n_constr=n_constr,
            lower=np.asarray(lower, dtype=float),
            upper=np.asarray(upper, dtype=float),
        )
        self._n_evals = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def n_var(self) -> int:
        return self.spec.n_var

    @property
    def n_obj(self) -> int:
        return self.spec.n_obj

    @property
    def n_constr(self) -> int:
        return self.spec.n_constr

    @property
    def lower(self) -> np.ndarray:
        return self.spec.lower

    @property
    def upper(self) -> np.ndarray:
        return self.spec.upper

    @property
    def n_evals(self) -> int:
        return self._n_evals

    def reset_eval_counter(self) -> None:
        self._n_evals = 0

    def evaluate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of solutions.

        Parameters
        ----------
        X : ndarray, shape (N, n_var)
            Decision vectors.

        Returns
        -------
        F : ndarray, shape (N, n_obj)
            Objective values (all minimised).
        CV : ndarray, shape (N,)
            Total constraint violation per solution (0.0 = feasible).
        """
        X = np.atleast_2d(X)
        # FIX(audit B10): ValueError instead of assert.
        if X.shape[1] != self.n_var:
            raise ValueError(
                f"X has {X.shape[1]} variables but problem expects {self.n_var}"
            )
        F, G = self._evaluate(X)
        self._n_evals += X.shape[0]

        # Aggregate constraint violation: sum of max(0, g_j) for each constraint
        if G is not None and G.shape[1] > 0:
            CV = np.sum(np.maximum(0.0, G), axis=1)
        else:
            CV = np.zeros(X.shape[0])
        return F, CV

    @abstractmethod
    def _evaluate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Evaluate objectives and raw constraint values.

        Returns
        -------
        F : ndarray, shape (N, n_obj)
        G : ndarray, shape (N, n_constr) or None
            Convention: g_j(x) <= 0 means feasible.
        """
        ...

    def pareto_front(self, n_points: int = 500) -> np.ndarray | None:
        """Return the analytical Pareto front, if known."""
        return None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_var={self.n_var}, "
            f"n_obj={self.n_obj}, n_constr={self.n_constr})"
        )
