"""NSGA-II: Non-dominated Sorting Genetic Algorithm II.

Faithful to Deb et al. (2002) with a feasibility-first constraint rule so it
can also run on constrained benchmark problems.

This is a correct, minimal baseline for sanity-checking MOSADE's results.
It is not competition-grade (no restarts, no adaptive operators), but it
implements all four canonical NSGA-II mechanisms:

  1. SBX crossover.
  2. Polynomial mutation (shared with MOSADE from strategies.py).
  3. Fast non-dominated sorting with feasibility-first comparison.
  4. Binary tournament selection on rank + crowding distance.

Usage::

    from mosade.algorithm.nsga2 import NSGA2
    from mosade.problems import ZDT1

    result = NSGA2(pop_size=100, max_evals=100_000, seed=42).run(ZDT1())
    # result is a MOSADEResult (same API as MOSADE)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mosade.algorithm.mosade import MOSADEResult
from mosade.algorithm.strategies import polynomial_mutation
from mosade.metrics.hypervolume import hypervolume
from mosade.metrics.igd import igd as compute_igd
from mosade.problems.base import Problem
from mosade.utils.seeding import get_rng

log = logging.getLogger("mosade.algorithm")


# ---------------------------------------------------------------------------
# Public algorithm class
# ---------------------------------------------------------------------------


class NSGA2:
    """Non-dominated Sorting Genetic Algorithm II (Deb et al., 2002).

    Parameters
    ----------
    pop_size : int
        Population size N.  Rounded up to the nearest even number so that
        paired SBX crossover produces exactly N offspring.
    max_evals : int
        Total function evaluation budget.
    seed : int or None
        Random seed for reproducibility.
    crossover_prob : float
        Probability that a parent pair undergoes SBX crossover.  Variables
        within a selected pair are each crossed with probability 0.5.
    crossover_eta : float
        SBX distribution index eta_c (higher → offspring closer to parents).
    mutation_eta : float
        Polynomial mutation distribution index eta_m.
    track_interval : int
        Evaluation interval for optional HV/IGD convergence snapshots.
    """

    def __init__(
        self,
        pop_size: int = 100,
        max_evals: int = 100_000,
        seed: int | None = None,
        crossover_prob: float = 0.9,
        crossover_eta: float = 20.0,
        mutation_eta: float = 20.0,
        track_interval: int = 0,
        **_extra: Any,  # absorb unknown keys from YAML configs gracefully
    ) -> None:
        # pop_size must be even so that paired crossover produces exactly N offspring.
        self.pop_size = pop_size if pop_size % 2 == 0 else pop_size + 1
        self.max_evals = max_evals
        self.seed = seed
        self.crossover_prob = crossover_prob
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta
        self.track_interval = track_interval

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        problem: Problem,
        pf: np.ndarray | None = None,
        ref_point: np.ndarray | None = None,
    ) -> MOSADEResult:
        """Execute NSGA-II on *problem*.

        Parameters
        ----------
        problem : Problem
            The optimisation problem to solve.
        pf : ndarray or None
            True Pareto front.  NSGA-II does not use this internally; the
            parameter exists for API compatibility with the experiment runner.
        ref_point : ndarray or None
            Ignored. Present for interface compatibility with the experiment
            runner and convergence-aware algorithms.

        Returns
        -------
        MOSADEResult
            Contains the final nondominated approximation set, decision vectors,
            total evaluation count, and a minimal history dict.
        """
        rng = get_rng(self.seed)
        N = self.pop_size
        D = problem.n_var

        problem.reset_eval_counter()

        # --- Initialise population ---
        X = rng.uniform(problem.lower, problem.upper, size=(N, D))
        F, CV = problem.evaluate(X)

        history: dict[str, list] = {"gen": [], "n_evals": []}
        if self.track_interval > 0:
            history["convergence"] = []
        _last_track_evals = 0
        gen = 0

        while problem.n_evals < self.max_evals:
            gen += 1

            # Compute rank and crowding once per generation (reused for selection).
            ranks = _fast_nondominated_sort(F, CV)
            cd = _crowding_distance_all(F, ranks)

            # Binary tournament selection → two parent permutations of size N.
            p_a = _tournament_select(ranks, cd, N, rng)
            p_b = _tournament_select(ranks, cd, N, rng)

            # SBX crossover + polynomial mutation → N offspring.
            Q = np.empty_like(X)
            for i in range(0, N, 2):
                c1, c2 = _sbx_crossover(
                    X[p_a[i]], X[p_b[i]],
                    problem.lower, problem.upper,
                    self.crossover_eta, rng, self.crossover_prob,
                )
                c1 = polynomial_mutation(
                    c1, problem.lower, problem.upper, rng, eta_m=self.mutation_eta
                )
                Q[i] = c1
                # The final slot when N is odd (can't happen after rounding, but guard anyway).
                if i + 1 < N:
                    c2 = polynomial_mutation(
                        c2, problem.lower, problem.upper, rng, eta_m=self.mutation_eta
                    )
                    Q[i + 1] = c2

            # Evaluate offspring.
            F_Q, CV_Q = problem.evaluate(Q)

            # Merge parent + offspring (2N pool), then select N survivors.
            X_all = np.vstack([X, Q])
            F_all = np.vstack([F, F_Q])
            CV_all = np.concatenate([CV, CV_Q])

            X, F, CV = _environmental_selection(X_all, F_all, CV_all, N)

            history["gen"].append(gen)
            history["n_evals"].append(problem.n_evals)
            if self.track_interval > 0 and (
                problem.n_evals - _last_track_evals >= self.track_interval
            ):
                history["convergence"].append(
                    _convergence_snapshot(F, CV, problem.n_evals, pf, ref_point)
                )
                _last_track_evals = problem.n_evals

        # --- Return the final nondominated feasible front ---
        # Prefer strictly feasible solutions; fall back to all if none are feasible.
        feas_mask = CV <= 0.0
        if feas_mask.any():
            F_out, X_out = F[feas_mask], X[feas_mask]
        else:
            F_out, X_out = F.copy(), X.copy()

        nd_ranks = _fast_nondominated_sort(F_out, np.zeros(F_out.shape[0]))
        nd_mask = nd_ranks == 0
        out_F = F_out[nd_mask] if nd_mask.any() else F_out
        out_X = X_out[nd_mask] if nd_mask.any() else X_out

        return MOSADEResult(
            X=out_X,
            F=out_F,
            n_evals=problem.n_evals,
            history=history,
            metadata={"pf_source": "final_population_feasible_nondominated"},
        )


# ---------------------------------------------------------------------------
# Non-dominated sorting
# ---------------------------------------------------------------------------


def _cdom(
    f_a: np.ndarray, cv_a: float, f_b: np.ndarray, cv_b: float
) -> bool:
    """Feasibility-first constrained domination (returns True if a ≻ b).

    Rules (following Deb et al.'s constraint-handling extension):
      - Feasible solution dominates any infeasible one.
      - Two infeasible: the one with lower CV is preferred (strict inequality).
      - Two feasible: standard Pareto dominance.
    """
    a_feas = cv_a <= 0.0
    b_feas = cv_b <= 0.0
    if a_feas and not b_feas:
        return True
    if not a_feas and b_feas:
        return False
    if not a_feas:  # both infeasible
        return bool(cv_a < cv_b)
    # Both feasible: standard Pareto dominance.
    return bool(np.all(f_a <= f_b) and np.any(f_a < f_b))


def _fast_nondominated_sort(F: np.ndarray, CV: np.ndarray) -> np.ndarray:
    """Assign a non-domination rank to each solution.

    Parameters
    ----------
    F : ndarray, shape (N, M)
    CV : ndarray, shape (N,)
        Total constraint violation (0.0 = feasible).

    Returns
    -------
    ranks : ndarray of int, shape (N,)
        Rank 0 = non-dominated (Pareto) front; rank 1 = dominated only by
        front 0; etc.

    Notes
    -----
    Time complexity O(N² M).  Adequate for populations up to a few hundred.
    """
    N = F.shape[0]
    # dom_count[i]: number of solutions that dominate solution i.
    dom_count = np.zeros(N, dtype=np.intp)
    # dom_by[i]: list of solutions that i dominates.
    dom_by: list[list[int]] = [[] for _ in range(N)]

    for i in range(N):
        for j in range(i + 1, N):
            if _cdom(F[i], CV[i], F[j], CV[j]):
                dom_by[i].append(j)
                dom_count[j] += 1
            elif _cdom(F[j], CV[j], F[i], CV[i]):
                dom_by[j].append(i)
                dom_count[i] += 1

    ranks = np.empty(N, dtype=np.intp)
    current_front = [i for i in range(N) if dom_count[i] == 0]
    rank = 0
    while current_front:
        for i in current_front:
            ranks[i] = rank
        next_front: list[int] = []
        for i in current_front:
            for j in dom_by[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    next_front.append(j)
        current_front = next_front
        rank += 1

    return ranks


# ---------------------------------------------------------------------------
# Crowding distance
# ---------------------------------------------------------------------------


def _crowding_distance_all(F: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Compute crowding distance for every solution.

    Boundary solutions in each front receive infinite distance so they are
    always preferred in tournaments (preserves objective-space coverage).

    Parameters
    ----------
    F : ndarray, shape (N, M)
    ranks : ndarray of int, shape (N,)

    Returns
    -------
    cd : ndarray of float, shape (N,)
    """
    N = F.shape[0]
    cd = np.zeros(N, dtype=float)
    n_ranks = int(ranks.max()) + 1 if N > 0 else 0

    for r in range(n_ranks):
        front = np.where(ranks == r)[0]
        n_front = len(front)
        if n_front <= 2:
            cd[front] = np.inf
            continue

        # Accumulate crowding contribution per objective.
        M = F.shape[1]
        for m in range(M):
            order = np.argsort(F[front, m])
            f_sorted = F[front[order], m]
            f_min, f_max = f_sorted[0], f_sorted[-1]
            span = f_max - f_min

            # Boundary points always get infinity.
            cd[front[order[0]]] = np.inf
            cd[front[order[-1]]] = np.inf

            if span < 1e-12:
                continue

            # Interior points.
            interior = order[1:-1]
            cd[front[interior]] += (
                F[front[order[2:]], m] - F[front[order[:-2]], m]
            ) / span

    return cd


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _tournament_select(
    ranks: np.ndarray, cd: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Return *n* parent indices via binary tournament (rank ≻ crowding).

    Two random individuals are drawn; the one with lower rank wins.
    Ties in rank are broken by higher crowding distance.

    Parameters
    ----------
    ranks : ndarray of int, shape (N,)
    cd : ndarray of float, shape (N,)
    n : int
        Number of parents to select.
    rng : Generator

    Returns
    -------
    winners : ndarray of int, shape (n,)
    """
    N = len(ranks)
    a = rng.integers(0, N, size=n)
    b = rng.integers(0, N, size=n)

    # a beats b when: rank_a < rank_b, or (rank_a == rank_b and cd_a > cd_b).
    a_wins = np.where(ranks[a] != ranks[b], ranks[a] < ranks[b], cd[a] > cd[b])
    return np.where(a_wins, a, b)


# ---------------------------------------------------------------------------
# Environmental selection
# ---------------------------------------------------------------------------


def _environmental_selection(
    X_all: np.ndarray,
    F_all: np.ndarray,
    CV_all: np.ndarray,
    N: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select *N* survivors from a combined 2N pool using NSGA-II front filling.

    Fronts are added in order of rank until the next front would overflow.
    The overflowing front is sorted by crowding distance (descending) and
    the top-k individuals needed to reach exactly N are kept.

    Parameters
    ----------
    X_all : ndarray, shape (2N, D)
    F_all : ndarray, shape (2N, M)
    CV_all : ndarray, shape (2N,)
    N : int
        Desired survivor count.

    Returns
    -------
    X, F, CV : ndarrays of shape (N, D), (N, M), (N,)
    """
    ranks = _fast_nondominated_sort(F_all, CV_all)
    cd = _crowding_distance_all(F_all, ranks)

    keep = np.zeros(F_all.shape[0], dtype=bool)
    n_kept = 0

    for r in range(int(ranks.max()) + 1):
        front = np.where(ranks == r)[0]
        if n_kept + len(front) <= N:
            keep[front] = True
            n_kept += len(front)
            if n_kept == N:
                break
        else:
            # Partial front: take the top-k by crowding distance.
            k = N - n_kept
            # Descending crowding distance.
            top_k = front[np.argsort(-cd[front])[:k]]
            keep[top_k] = True
            break

    return X_all[keep], F_all[keep], CV_all[keep]


def _convergence_snapshot(
    F: np.ndarray,
    CV: np.ndarray,
    n_evals: int,
    pf: np.ndarray | None,
    ref_point: np.ndarray | None,
) -> dict[str, object]:
    """Compute HV/IGD convergence metrics from the live NSGA-II population."""
    feasible = F[CV <= 0.0]
    if feasible.size == 0:
        F_nd = np.empty((0, F.shape[1]), dtype=float)
    else:
        ranks = _fast_nondominated_sort(feasible, np.zeros(feasible.shape[0]))
        F_nd = feasible[ranks == 0]

    hv_val: float | None = 0.0
    if ref_point is not None and F_nd.shape[0] > 0:
        hv_val = float(hypervolume(F_nd, np.asarray(ref_point, dtype=float)))

    igd_val: float | None = None
    if pf is not None and F_nd.shape[0] > 0:
        igd_val = float(compute_igd(F_nd, pf))

    return {"n_evals": int(n_evals), "hv": hv_val, "igd": igd_val}


# ---------------------------------------------------------------------------
# SBX crossover
# ---------------------------------------------------------------------------


def _sbx_crossover(
    x1: np.ndarray,
    x2: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eta_c: float,
    rng: np.random.Generator,
    prob: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover (SBX, Deb & Agrawal 1995).

    Produces two children from two parents while respecting variable bounds.
    Each variable is independently crossed with probability 0.5 when the pair
    is selected for crossover (controlled by *prob*).

    Parameters
    ----------
    x1, x2 : ndarray, shape (D,)
        Parent decision vectors.
    lower, upper : ndarray, shape (D,)
        Variable bounds.
    eta_c : float
        Distribution index.  Larger values concentrate offspring near parents.
    rng : Generator
    prob : float
        Probability the parent pair is crossed at all.

    Returns
    -------
    c1, c2 : two child decision vectors, shape (D,)
    """
    c1, c2 = x1.copy(), x2.copy()

    if rng.random() > prob:
        return c1, c2

    D = len(x1)
    for j in range(D):
        # Each gene crossed with prob 0.5.
        if rng.random() > 0.5:
            continue

        y1 = min(x1[j], x2[j])
        y2 = max(x1[j], x2[j])
        lb, ub = lower[j], upper[j]

        if y2 - y1 < 1e-14:
            continue

        # Spread factor (bounded by distance to nearest boundary).
        beta = 1.0 + 2.0 * min(y1 - lb, ub - y2) / (y2 - y1)
        alpha = 2.0 - beta ** (-(eta_c + 1.0))

        u = rng.random()
        if u <= 1.0 / alpha:
            beta_q = (u * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            beta_q = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))

        # Two offspring values (symmetric around midpoint).
        c1j = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
        c2j = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))

        # Clip to bounds.
        c1j = float(np.clip(c1j, lb, ub))
        c2j = float(np.clip(c2j, lb, ub))

        # Preserve original ordering: x1[j] ≤ x2[j] → c1 gets the smaller child.
        if x1[j] <= x2[j]:
            c1[j], c2[j] = c1j, c2j
        else:
            c1[j], c2[j] = c2j, c1j

    return c1, c2
