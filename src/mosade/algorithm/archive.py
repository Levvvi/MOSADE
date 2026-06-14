"""External archive: passive elite storage with crowding-based truncation."""

from __future__ import annotations

import numpy as np

from mosade.algorithm.selection import dominates


class Archive:
    """Bounded external archive of nondominated feasible solutions.

    Used for:
      - Providing pbest candidates to mutation strategies S1 and S4.
      - Final output of the algorithm.

    Truncation is by crowding distance (removing least-crowded member).
    """

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.X: list[np.ndarray] = []
        self.F: list[np.ndarray] = []

    @property
    def size(self) -> int:
        return len(self.X)

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    def get_objectives(self) -> np.ndarray:
        """Return objective matrix, shape (size, M)."""
        if self.is_empty:
            raise ValueError("Archive is empty.")
        return np.array(self.F)

    def get_decisions(self) -> np.ndarray:
        """Return decision matrix, shape (size, D)."""
        if self.is_empty:
            raise ValueError("Archive is empty.")
        return np.array(self.X)

    def update(self, X_new: np.ndarray, F_new: np.ndarray, CV_new: np.ndarray) -> None:
        """Add feasible nondominated solutions and truncate if needed.

        Parameters
        ----------
        X_new : ndarray, shape (K, D)
        F_new : ndarray, shape (K, M)
        CV_new : ndarray, shape (K,)
        """
        # Only consider feasible solutions
        feas = CV_new <= 0.0
        if not np.any(feas):
            return

        X_cand = X_new[feas]
        F_cand = F_new[feas]

        # Skip candidates with non-finite objectives or decision variables.
        finite = np.all(np.isfinite(F_cand), axis=1) & np.all(np.isfinite(X_cand), axis=1)
        X_cand = X_cand[finite]
        F_cand = F_cand[finite]
        if len(X_cand) == 0:
            return

        for i in range(len(X_cand)):
            # Check if candidate is dominated by any archive member
            dominated_by_archive = False
            to_remove = []
            for j in range(len(self.F)):
                if dominates(np.array(self.F[j]), F_cand[i]):
                    dominated_by_archive = True
                    break
                if dominates(F_cand[i], np.array(self.F[j])):
                    to_remove.append(j)

            if dominated_by_archive:
                continue

            # Remove dominated archive members (in reverse order to preserve indices)
            for j in sorted(to_remove, reverse=True):
                self.X.pop(j)
                self.F.pop(j)

            self.X.append(X_cand[i].copy())
            self.F.append(F_cand[i].copy())

        # Truncate by crowding distance
        while self.size > self.max_size:
            self._remove_least_crowded()

    def _remove_least_crowded(self) -> None:
        """Remove the archive member with smallest crowding distance."""
        F_arr = np.array(self.F)
        cd = _crowding_distance(F_arr)
        worst = int(np.argmin(cd))
        self.X.pop(worst)
        self.F.pop(worst)

    def random_member(self, rng: np.random.Generator) -> np.ndarray:
        """Return decision vector of a uniformly random archive member."""
        idx = rng.integers(self.size)
        return np.array(self.X[idx])


def _crowding_distance(F: np.ndarray) -> np.ndarray:
    """Compute crowding distance for a set of objective vectors."""
    N, M = F.shape
    if N <= 2:
        return np.full(N, np.inf)

    cd = np.zeros(N)
    for m in range(M):
        order = np.argsort(F[:, m])
        cd[order[0]] = np.inf
        cd[order[-1]] = np.inf
        f_range = F[order[-1], m] - F[order[0], m]
        if f_range < 1e-30:
            continue
        for k in range(1, N - 1):
            cd[order[k]] += (F[order[k + 1], m] - F[order[k - 1], m]) / f_range
    return cd
