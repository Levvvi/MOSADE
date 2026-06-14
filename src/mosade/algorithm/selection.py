"""Environmental selection: epsilon-constraint handling, dominance utilities,
and per-subproblem selection for the decomposition framework.
"""

from __future__ import annotations

import numpy as np

from mosade.algorithm.decomposition import tchebycheff


# ---------------------------------------------------------------------------
# Dominance utilities
# ---------------------------------------------------------------------------


def dominates(f_a: np.ndarray, f_b: np.ndarray) -> bool:
    """Return True if f_a Pareto-dominates f_b (all minimised)."""
    return bool(np.all(f_a <= f_b) and np.any(f_a < f_b))


def nondominated_mask(F: np.ndarray) -> np.ndarray:
    """Return a boolean mask of non-dominated solutions.

    Parameters
    ----------
    F : ndarray, shape (N, M)

    Returns
    -------
    mask : ndarray of bool, shape (N,)
    """
    N = F.shape[0]
    is_nd = np.ones(N, dtype=bool)
    for i in range(N):
        if not is_nd[i]:
            continue
        for j in range(i + 1, N):
            if not is_nd[j]:
                continue
            if dominates(F[i], F[j]):
                is_nd[j] = False
            elif dominates(F[j], F[i]):
                is_nd[i] = False
                break
    return is_nd


def constrained_nondominated_sort(F: np.ndarray, CV: np.ndarray) -> np.ndarray:
    """Assign constrained non-domination ranks using feasible-first dominance.

    Rank 0 is the best front.  Feasible solutions dominate infeasible ones,
    feasible-feasible comparisons use Pareto dominance, and infeasible-infeasible
    comparisons prefer lower total constraint violation.
    """
    n = F.shape[0]
    dom_count = np.zeros(n, dtype=np.intp)
    dominates_list: list[list[int]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if constrained_dominates(F[i], CV[i], F[j], CV[j]):
                dominates_list[i].append(j)
                dom_count[j] += 1
            elif constrained_dominates(F[j], CV[j], F[i], CV[i]):
                dominates_list[j].append(i)
                dom_count[i] += 1

    ranks = np.full(n, -1, dtype=np.intp)
    current_front = [i for i in range(n) if dom_count[i] == 0]
    rank = 0
    while current_front:
        for i in current_front:
            ranks[i] = rank
        next_front: list[int] = []
        for i in current_front:
            for j in dominates_list[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    next_front.append(j)
        current_front = next_front
        rank += 1

    return ranks


def constrained_dominates(
    f_a: np.ndarray,
    cv_a: float,
    f_b: np.ndarray,
    cv_b: float,
) -> bool:
    """Return True when ``a`` dominates ``b`` under feasible-first rules."""
    a_feas = cv_a <= 0.0
    b_feas = cv_b <= 0.0
    if a_feas and not b_feas:
        return True
    if b_feas and not a_feas:
        return False
    if not a_feas and not b_feas:
        return bool(cv_a < cv_b)
    return dominates(f_a, f_b)


def crowding_distance_by_front(F: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Compute crowding distance for all points, independently per front."""
    n = F.shape[0]
    cd = np.zeros(n, dtype=float)
    if n == 0 or np.all(ranks < 0):
        return cd

    for rank in range(int(ranks.max()) + 1):
        front = np.where(ranks == rank)[0]
        n_front = len(front)
        if n_front <= 2:
            cd[front] = np.inf
            continue
        for m in range(F.shape[1]):
            order = np.argsort(F[front, m])
            f_sorted = F[front[order], m]
            span = f_sorted[-1] - f_sorted[0]
            cd[front[order[0]]] = np.inf
            cd[front[order[-1]]] = np.inf
            if span < 1e-12:
                continue
            interior = order[1:-1]
            cd[front[interior]] += (
                F[front[order[2:]], m] - F[front[order[:-2]], m]
            ) / span
    return cd


def select_dominance_survival(
    F_all: np.ndarray,
    CV_all: np.ndarray,
    n_select: int,
) -> tuple[np.ndarray, dict[str, int | str]]:
    """Select survivors by constrained dominance and crowding truncation.

    Returns selected indices into the input pool and compact instrumentation
    metadata used by MOSADE ablation runs.
    """
    ranks = constrained_nondominated_sort(F_all, CV_all)
    cd = crowding_distance_by_front(F_all, ranks)
    selected: list[int] = []
    tie_break_count = 0
    truncation_method = "none"

    for rank in range(int(ranks.max()) + 1 if ranks.size else 0):
        front = np.where(ranks == rank)[0]
        if len(selected) + len(front) <= n_select:
            selected.extend(int(i) for i in front)
            if len(selected) == n_select:
                break
            continue

        remaining = n_select - len(selected)
        order = np.argsort(-cd[front], kind="mergesort")
        selected.extend(int(i) for i in front[order[:remaining]])
        tie_break_count += int(len(front) - remaining)
        truncation_method = "crowding_distance"
        break

    if len(selected) < n_select:
        remaining = [i for i in range(F_all.shape[0]) if i not in set(selected)]
        order = np.lexsort((F_all[remaining].sum(axis=1), CV_all[remaining]))
        selected.extend(int(remaining[i]) for i in order[: n_select - len(selected)])
        truncation_method = "fallback_cv_sum"

    selected_array = np.array(selected[:n_select], dtype=int)
    meta = {
        "n_fronts": int(ranks.max() + 1) if ranks.size else 0,
        "truncation_method": truncation_method,
        "selection_tie_break_count": int(tie_break_count),
    }
    return selected_array, meta


# ---------------------------------------------------------------------------
# Epsilon-constraint handling
# ---------------------------------------------------------------------------


class EpsilonConstraint:
    """Adaptive epsilon-constraint relaxation.

    Starts permissive and tightens to zero over the schedule.

    Parameters
    ----------
    theta : float
        Percentile (0-1) of initial CV distribution used for epsilon_0.
    T_c_ratio : float
        Fraction of total generations at which epsilon reaches 0.
    cp : float
        Decay power (higher = sharper tightening).
    """

    def __init__(
        self, theta: float = 0.8, T_c_ratio: float = 0.8, cp: float = 2.0
    ) -> None:
        self.theta = theta
        self.T_c_ratio = T_c_ratio
        self.cp = cp
        self.epsilon_0: float = 0.0
        self.T_c: int = 1
        self._initialised = False

    def initialise(self, CV: np.ndarray, max_gen: int) -> None:
        """Set epsilon_0 from initial population constraint violations."""
        self.T_c = max(1, int(self.T_c_ratio * max_gen))
        if np.any(CV > 0):
            self.epsilon_0 = float(np.percentile(CV, self.theta * 100))
        else:
            self.epsilon_0 = 0.0
        self._initialised = True

    def __call__(self, gen: int) -> float:
        """Return epsilon value at generation *gen*."""
        if not self._initialised or self.epsilon_0 == 0.0:
            return 0.0
        ratio = min(gen / self.T_c, 1.0)
        return self.epsilon_0 * max(0.0, 1.0 - ratio) ** self.cp


def epsilon_feasibility_compare(
    cv_a: float,
    cv_b: float,
    g_a: float,
    g_b: float,
    dom_a_b: bool,
    dom_b_a: bool,
    epsilon: float,
) -> int:
    """Compare two solutions under epsilon-feasibility rules.

    Returns
    -------
    -1 if a is preferred, +1 if b is preferred, 0 if tie.
    """
    a_feas = cv_a <= epsilon
    b_feas = cv_b <= epsilon

    if a_feas and not b_feas:
        return -1
    if b_feas and not a_feas:
        return 1
    if not a_feas and not b_feas:
        # Both infeasible: prefer smaller CV
        if cv_a < cv_b:
            return -1
        elif cv_b < cv_a:
            return 1
        return 0

    # Both epsilon-feasible: dominance, then scalarising value
    if dom_a_b:
        return -1
    if dom_b_a:
        return 1
    if g_a < g_b:
        return -1
    if g_b < g_a:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Per-subproblem environmental selection
# ---------------------------------------------------------------------------


def select_per_subproblem(
    F_all: np.ndarray,
    CV_all: np.ndarray,
    assoc: np.ndarray,
    weights: np.ndarray,
    z_ideal: np.ndarray,
    epsilon: float,
    n_weights: int,
    neighbors: np.ndarray,
) -> np.ndarray:
    """Select one solution per weight vector from a combined pool.

    Parameters
    ----------
    F_all : ndarray, shape (2N, M)
    CV_all : ndarray, shape (2N,)
    assoc : ndarray, shape (2N,)
        Weight vector association index per solution.
    weights : ndarray, shape (W, M)
    z_ideal : ndarray, shape (M,)
    epsilon : float
        Current epsilon threshold.
    n_weights : int
        Number of weight vectors (= population size N).
    neighbors : ndarray, shape (W, T)
        Neighborhood index array.

    Returns
    -------
    selected : ndarray, shape (N,)
        Indices into the combined pool.
    """
    selected = np.full(n_weights, -1, dtype=int)
    used = set()

    for j in range(n_weights):
        # Candidates associated with this weight vector
        cands = np.where(assoc == j)[0]
        if len(cands) == 0:
            # Borrow from neighboring subproblems
            for nb in neighbors[j]:
                nb_cands = np.where(assoc == nb)[0]
                nb_cands = np.array([c for c in nb_cands if c not in used])
                if len(nb_cands) > 0:
                    cands = nb_cands
                    break

        if len(cands) == 0:
            continue  # will be filled in a fallback pass

        # Pick best candidate
        best = cands[0]
        g_best = tchebycheff(F_all[best], weights[j], z_ideal)
        for c in cands[1:]:
            g_c = tchebycheff(F_all[c], weights[j], z_ideal)
            dom_best_c = dominates(F_all[best], F_all[c])
            dom_c_best = dominates(F_all[c], F_all[best])
            cmp = epsilon_feasibility_compare(
                CV_all[best], CV_all[c], g_best, g_c,
                dom_best_c, dom_c_best, epsilon,
            )
            if cmp > 0:  # c is preferred
                best = c
                g_best = g_c
        selected[j] = best
        used.add(best)

    # Fallback: fill any -1 entries with the globally best remaining solutions
    unfilled = np.where(selected == -1)[0]
    if len(unfilled) > 0:
        remaining = np.array([i for i in range(F_all.shape[0]) if i not in used])
        if len(remaining) > 0:
            # Sort remaining by CV then by sum of objectives (rough heuristic)
            order = np.lexsort((F_all[remaining].sum(axis=1), CV_all[remaining]))
            remaining_sorted = remaining[order]
            for idx, j in enumerate(unfilled):
                if idx < len(remaining_sorted):
                    selected[j] = remaining_sorted[idx]
                    used.add(remaining_sorted[idx])

    return selected
