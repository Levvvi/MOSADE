"""MOEA/D: Multi-Objective Evolutionary Algorithm Based on Decomposition.

Faithful to Zhang & Li (2007) with the DE reproduction operator from
Li & Zhang (2009), plus a feasibility-first constraint rule so it works
on constrained benchmark problems without modification.

Mechanisms
----------
1. Das-Dennis weight vectors (reused from :mod:`decomposition`).
2. T-nearest neighbourhood in weight space (reused from :mod:`decomposition`).
3. Tchebycheff scalarisation (reused from :mod:`decomposition`).
4. DE/rand/1 mutation + binomial crossover within the mating pool.
5. Polynomial mutation (reused from :mod:`strategies`).
6. Online neighbourhood replacement: each offspring updates all neighbours
   whose Tchebycheff value it improves.
7. Feasibility-first comparison: feasible beats infeasible; infeasible
   compared by CV; feasible compared by Tchebycheff value.

Usage::

    from mosade.algorithm.moead import MOEAD
    from mosade.problems import ZDT1

    result = MOEAD(pop_size=100, max_evals=100_000, seed=42).run(ZDT1())
    # result is a MOSADEResult (same API as MOSADE and NSGA2)

References
----------
Zhang, Q. & Li, H. (2007). MOEA/D: A Multiobjective Evolutionary Algorithm
Based on Decomposition. IEEE TEVC, 11(6), 712–731.

Li, H. & Zhang, Q. (2009). Multiobjective Optimization Problems With
Complicated Pareto Sets, MOEA/D and NSGA-II. IEEE TEVC, 13(2), 284–302.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mosade.algorithm.decomposition import (
    auto_partitions,
    compute_neighbors,
    das_dennis,
    tchebycheff,
)
from mosade.algorithm.mosade import MOSADEResult
from mosade.algorithm.selection import nondominated_mask
from mosade.algorithm.strategies import (
    binomial_crossover,
    midpoint_repair,
    mutate_rand_1,
    polynomial_mutation,
)
from mosade.problems.base import Problem
from mosade.utils.seeding import get_rng

log = logging.getLogger("mosade.algorithm")


# ---------------------------------------------------------------------------
# Public algorithm class
# ---------------------------------------------------------------------------


class MOEAD:
    """Multi-Objective Evolutionary Algorithm Based on Decomposition (MOEA/D).

    Parameters
    ----------
    pop_size : int
        Requested population size N.  The effective size is adjusted to the
        nearest valid Das-Dennis weight-vector count, exactly as in MOSADE.
    max_evals : int
        Total function evaluation budget.
    seed : int or None
        Random seed for reproducibility.
    T_ratio : float
        Neighbourhood size T as a fraction of the effective N.
        T is clamped to at least 3 (so DE/rand/1 can always pick 3 distinct
        parents from the neighbourhood).  Default: 0.1.
    delta : float
        Probability that mating parents are drawn from the neighbourhood
        B(i) rather than the full population.  Matches the δ parameter in
        Zhang & Li (2007).  Default: 0.9.
    F : float
        DE/rand/1 scaling factor.  Default: 0.5.
    CR : float
        Binomial crossover rate.  Default: 1.0 (recommended for MOEA/D-DE).
    mutation_eta : float
        Polynomial mutation distribution index.  Default: 20.0.
    """

    def __init__(
        self,
        pop_size: int = 100,
        max_evals: int = 100_000,
        seed: int | None = None,
        T_ratio: float = 0.1,
        delta: float = 0.9,
        F: float = 0.5,
        CR: float = 1.0,
        mutation_eta: float = 20.0,
        **_extra: Any,  # absorb unknown YAML keys gracefully
    ) -> None:
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.seed = seed
        self.T_ratio = T_ratio
        self.delta = delta
        self.F = F
        self.CR = CR
        self.mutation_eta = mutation_eta

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        problem: Problem,
        pf: np.ndarray | None = None,
        ref_point: np.ndarray | None = None,
    ) -> MOSADEResult:
        """Execute MOEA/D on *problem*.

        Parameters
        ----------
        problem : Problem
            The optimisation problem to solve.
        pf : ndarray or None
            True Pareto front.  MOEA/D does not use this internally; the
            parameter exists for API compatibility with the experiment runner.
        ref_point : ndarray or None
            Ignored. Present for interface compatibility with the experiment
            runner and convergence-aware algorithms.

        Returns
        -------
        MOSADEResult
            Contains the final nondominated feasible approximation set,
            corresponding decision vectors, total evaluation count, and a
            minimal history dict with ``gen`` and ``n_evals`` lists.
        """
        rng = get_rng(self.seed)
        D = problem.n_var
        M = problem.n_obj

        problem.reset_eval_counter()

        # --- Weight vectors and neighbourhoods ---
        H = auto_partitions(self.pop_size, M)
        weights = das_dennis(H, M)
        N = weights.shape[0]
        log.info("Weight vectors: %d (H=%d), effective pop_size=%d", N, H, N)

        # T: neighbourhood size; clamped to at least 3 so DE/rand/1 always has
        # enough distinct parents to choose from within the neighbourhood.
        T = max(3, int(self.T_ratio * N))
        T = min(T, N - 1)
        neighbors = compute_neighbors(weights, T)

        # Pre-build neighbourhood arrays that include the subproblem itself
        # at position 0, giving a (N, T+1) update pool used in the online
        # replacement step.  Shape: (N, T+1).
        self_col = np.arange(N).reshape(-1, 1)
        update_pool = np.hstack([self_col, neighbors])  # i is always first

        # --- Initialise population ---
        X = rng.uniform(problem.lower, problem.upper, size=(N, D))
        F_pop, CV_pop = problem.evaluate(X)

        # Ideal point: running minimum over all evaluated objectives.
        z_ideal = F_pop.min(axis=0).copy()

        # --- Main loop ---
        history: dict[str, list] = {"gen": [], "n_evals": []}
        gen = 0

        while problem.n_evals < self.max_evals:
            gen += 1

            for i in range(N):
                # Hard budget check inside inner loop: MOEA/D evaluates one
                # solution per subproblem per generation, so we must check
                # here to avoid overshooting max_evals by up to N-1 evals.
                if problem.n_evals >= self.max_evals:
                    break

                # ----------------------------------------------------------
                # Step 1: Mating selection
                # With probability δ, draw from the neighbourhood B(i).
                # Otherwise draw from the entire population.
                # ----------------------------------------------------------
                if rng.random() < self.delta:
                    pool = neighbors[i]       # shape (T,), never contains i
                else:
                    pool = np.arange(N)       # full population

                r1, r2, r3 = _sample_three(pool, i, N, rng)

                # ----------------------------------------------------------
                # Step 2: Reproduction — DE/rand/1 + binomial crossover
                # v = x_{r1} + F*(x_{r2} - x_{r3})
                # u = binomial_crossover(x_i, v, CR)
                # ----------------------------------------------------------
                v = mutate_rand_1(X[r1], X[r2], X[r3], self.F)
                u = binomial_crossover(X[i], v, self.CR, rng)

                # Midpoint repair (handles DE overshoot) then polynomial mutation.
                u = midpoint_repair(u, X[i], problem.lower, problem.upper)
                u = polynomial_mutation(
                    u, problem.lower, problem.upper, rng,
                    eta_m=self.mutation_eta,
                )

                # ----------------------------------------------------------
                # Step 3: Evaluate offspring
                # ----------------------------------------------------------
                f_u_batch, cv_u_batch = problem.evaluate(u.reshape(1, -1))
                f_u: np.ndarray = f_u_batch[0]
                cv_u: float = float(cv_u_batch[0])

                # ----------------------------------------------------------
                # Step 4: Update ideal point
                # Done before neighbourhood update so the Tchebycheff
                # comparison uses the tightest available reference.
                # ----------------------------------------------------------
                z_ideal = np.minimum(z_ideal, f_u)

                # ----------------------------------------------------------
                # Step 5: Online neighbourhood replacement
                # For every j in {i} ∪ B(i), replace x^j with the offspring
                # if the offspring is preferred under λ^j.
                # ----------------------------------------------------------
                for j in update_pool[i]:
                    if _preferred(
                        f_u, cv_u,
                        F_pop[j], CV_pop[j],
                        weights[j], z_ideal,
                    ):
                        X[j] = u.copy()
                        F_pop[j] = f_u.copy()
                        CV_pop[j] = cv_u

            history["gen"].append(gen)
            history["n_evals"].append(problem.n_evals)

        # --- Extract final nondominated feasible front ---
        feas_mask = CV_pop <= 0.0
        if feas_mask.any():
            F_out = F_pop[feas_mask]
            X_out = X[feas_mask]
        else:
            # No strictly feasible solutions (constrained problem with tight
            # constraints): return the full population and let the caller handle it.
            F_out = F_pop.copy()
            X_out = X.copy()

        nd_mask = nondominated_mask(F_out)
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
# Module-level helpers
# ---------------------------------------------------------------------------


def _preferred(
    f_new: np.ndarray,
    cv_new: float,
    f_old: np.ndarray,
    cv_old: float,
    weight: np.ndarray,
    z_ideal: np.ndarray,
) -> bool:
    """Return True if *new* is preferred over *old* for the given subproblem.

    Comparison rules (feasibility-first):

    1. Feasible beats infeasible.
    2. Two infeasible: lower CV is preferred.
    3. Two feasible: lower Tchebycheff scalarisation value is preferred
       (``<=`` so equal-quality offspring can replace, improving diversity).

    Parameters
    ----------
    f_new, f_old : ndarray, shape (M,)
        Objective vectors.
    cv_new, cv_old : float
        Total constraint violations (0.0 = feasible).
    weight : ndarray, shape (M,)
        Weight vector for this subproblem.
    z_ideal : ndarray, shape (M,)
        Current ideal point.
    """
    new_feas = cv_new <= 0.0
    old_feas = cv_old <= 0.0

    if new_feas and not old_feas:
        return True
    if not new_feas and old_feas:
        return False
    if not new_feas:  # both infeasible: prefer smaller violation
        return bool(cv_new < cv_old)

    # Both feasible: Tchebycheff comparison.
    g_new = float(tchebycheff(f_new, weight, z_ideal))
    g_old = float(tchebycheff(f_old, weight, z_ideal))
    return g_new <= g_old


def _sample_three(
    pool: np.ndarray,
    exclude: int,
    N: int,
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    """Sample 3 distinct parent indices from *pool*, excluding *exclude*.

    Falls back to the full population (range(N) minus *exclude*) if the pool
    has fewer than 3 eligible candidates.

    Parameters
    ----------
    pool : ndarray of int
        Candidate indices (e.g. neighbourhood or np.arange(N)).
    exclude : int
        Index of the current subproblem (must not be selected as a parent).
    N : int
        Total population size (used for the fallback).
    rng : Generator

    Returns
    -------
    r1, r2, r3 : three distinct int indices, none equal to *exclude*
    """
    cands = pool[pool != exclude]
    if cands.size < 3:
        # Neighbourhood too small — fall back to global population.
        cands = np.array([k for k in range(N) if k != exclude])
    chosen = rng.choice(len(cands), size=3, replace=False)
    return int(cands[chosen[0]]), int(cands[chosen[1]]), int(cands[chosen[2]])
