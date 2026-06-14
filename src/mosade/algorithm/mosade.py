"""MOSADE: Multi-Objective Self-Adaptive Differential Evolution.

This is the main algorithm class.  See the design document for full
specification of each mechanism.

STATUS:
  - Decomposition, strategies, LSHADE memories, credit assignment,
    epsilon-constraint handling, archive, restart: IMPLEMENTED.
  - Adaptive neighbourhood size T: IMPLEMENTED (simplified).
  - The algorithm is runnable end-to-end on unconstrained and
    constrained problems.

ASSUMPTIONS (recorded for future development):
  A1. Polynomial mutation probability defaults to 1/D, eta_m=20.
  A2. Archive size is capped at 2*N.
  A3. Stagnation restart threshold L_stag = 0.1 * max_gen.
  A4. For bi-objective, Das-Dennis H is chosen automatically so that
      the number of weight vectors is close to N.
"""

from __future__ import annotations

import logging
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from mosade.algorithm.adaptation import LSHADEMemory, StrategySelector
from mosade.algorithm.archive import Archive
from mosade.algorithm.decomposition import (
    associate_to_weights,
    auto_partitions,
    compute_neighbors,
    das_dennis,
    tchebycheff,
)
from mosade.algorithm.selection import (
    EpsilonConstraint,
    constrained_nondominated_sort,
    crowding_distance_by_front,
    dominates,
    nondominated_mask,
    select_dominance_survival,
    select_per_subproblem,
)
from mosade.algorithm.strategies import (
    NUM_STRATEGIES,
    Strategy,
    binomial_crossover,
    midpoint_repair,
    mutate_current_to_pbest_1,
    mutate_current_to_rand_1,
    mutate_rand_1,
    mutate_rand_to_pbest_2,
    polynomial_mutation,
)
from mosade.metrics.hypervolume import hypervolume
from mosade.metrics.igd import igd as compute_igd
from mosade.problems.base import Problem
from mosade.utils.seeding import get_rng

log = logging.getLogger("mosade.algorithm")

EpsilonMode = Literal["adaptive", "fixed_initial", "zero"]
MemoryScope = Literal["per_strategy", "shared"]
SelectionMode = Literal["decomposition", "dominance"]
SuccessCriterion = Literal[
    "current",
    "dominance",
    "rank_crowding",
    "decomposition",
    "feasibility_first",
    "hvproxy",
]

EPS_MODES: frozenset[str] = frozenset({"adaptive", "fixed_initial", "zero"})
MEMORY_SCOPES: frozenset[str] = frozenset({"per_strategy", "shared"})
SELECTION_MODES: frozenset[str] = frozenset({"decomposition", "dominance"})
SUCCESS_CRITERIA: frozenset[str] = frozenset({
    "current",
    "dominance",
    "rank_crowding",
    "decomposition",
    "feasibility_first",
    "hvproxy",
})
TELEMETRY_SCHEMA_VERSION = "sr_diagnosis_v1"


# ---------------------------------------------------------------------------
# Public result container
# ---------------------------------------------------------------------------

@dataclass
class MOSADEResult:
    """Container for a single algorithm run's output."""

    X: np.ndarray  # decision vectors, shape (N_final, D)
    F: np.ndarray  # objective vectors, shape (N_final, M)
    n_evals: int
    history: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    message: str | None = None


# ---------------------------------------------------------------------------
# Internal per-generation state bundle
# ---------------------------------------------------------------------------

@dataclass
class _GenState:
    """All mutable per-generation algorithm state in one place.

    Passed by reference to offspring-generation helpers so that each helper
    receives a single ``state`` argument instead of up to eleven individual
    arrays.  This eliminates parameter confusion between indices and vectors
    (e.g. accidentally passing a population index where a decision vector is
    expected, or vice-versa).

    Fields
    ------
    X : ndarray, shape (N, D)
        Decision matrix.  Mutated in-place by the selection step.
    F : ndarray, shape (N, M)
        Objective matrix.  Mutated in-place by the selection step.
    CV : ndarray, shape (N,)
        Constraint violations.  Mutated in-place by the selection step.
    assoc : ndarray, shape (N,)
        Weight-vector association index for each individual.
    z_ideal : ndarray, shape (M,)
        Running ideal point (non-increasing per objective).
    z_nadir : ndarray, shape (M,)
        Running nadir point (non-decreasing per objective).
    weights : ndarray, shape (W, M)
        Weight vectors (fixed for the run).
    neighbors : ndarray, shape (W, T)
        T-nearest weight-vector neighbours (may be replaced when T adapts).
    archive : Archive
        Passive external archive of nondominated feasible solutions.

    Notes
    -----
    ``z_ideal``, ``z_nadir``, ``neighbors``, and ``assoc`` are rebound to new
    arrays each generation by the adaptive mechanisms; callers must update
    the corresponding fields on the same ``_GenState`` instance.
    """

    X: np.ndarray
    F: np.ndarray
    CV: np.ndarray
    assoc: np.ndarray
    z_ideal: np.ndarray
    z_nadir: np.ndarray
    weights: np.ndarray
    neighbors: np.ndarray
    archive: Archive

    @property
    def N(self) -> int:
        """Effective population size (equals W after weight-vector adjustment)."""
        return self.X.shape[0]


# ---------------------------------------------------------------------------
# Main algorithm class
# ---------------------------------------------------------------------------

class MOSADE:
    """Multi-Objective Self-Adaptive Differential Evolution.

    Parameters
    ----------
    pop_size : int
        Population size N.
    max_evals : int
        Total function evaluation budget.
    seed : int or None
        Random seed for reproducibility.
    H : int or None
        Number of Das-Dennis partitions.  If None, chosen automatically
        from pop_size and n_obj.
    T_base_ratio : float
        Neighbourhood size as fraction of N.
    memory_H : int
        Number of entries in each LSHADE success memory.
    lp : int
        Sliding-window length for credit assignment.
    pi_min : float
        Minimum per-strategy selection probability (floor before re-normalisation).
    fixed_strategy : int or None
        If set, only use this strategy index (0–3) for all offspring.
        Disables strategy selection; credit assignment is skipped.
        Used for ablation studies.
    disable_credit : bool
        If True, keep strategy selection probabilities uniform (no credit
        assignment update).  All four strategies remain available.
        Used for ablation studies.
    disable_epsilon : bool
        Deprecated compatibility switch.  If True, maps to ``eps_mode="zero"``.
    eps_mode : {"adaptive", "fixed_initial", "zero"}
        Epsilon schedule used for constrained selection.  ``adaptive`` uses
        the decaying epsilon-constraint schedule, ``fixed_initial`` keeps the
        initial epsilon value for the whole run, and ``zero`` uses strict
        feasibility from generation 0.
    memory_scope : {"per_strategy", "shared"}
        Whether LSHADE success memories are strategy-local or shared by all
        strategies.
    restart_enabled : bool
        Whether stagnation-triggered restart is active.
    selection_mode : {"decomposition", "dominance"}
        Environmental survival rule.
    success_criterion : {"current", "dominance", "rank_crowding", "decomposition",
        "feasibility_first", "hvproxy"}
        Diagnostic-only credit/success definition used by adaptation updates.
        The default ``current`` preserves the original MOSADE behaviour.
    adapt_fcr : bool
        Whether successful offspring update the LSHADE F/CR memories and future
        F/CR values are sampled from those memories.
    adapt_by_success_rate : bool
        Whether success credit updates strategy-selection probabilities.
        When False, the selector remains at its initial uniform probabilities.
    fixed_F, fixed_CR : float
        Constants used when ``adapt_fcr`` is disabled.
    delta_min, delta_max : float
        Range for global/local mixing ratio schedule.
    stag_ratio : float
        Fraction of max generations used as stagnation patience.
    restart_keep : float
        Fraction of population retained during restart.
    """

    def __init__(
        self,
        pop_size: int = 100,
        max_evals: int = 100_000,
        seed: int | None = None,
        H: int | None = None,
        T_base_ratio: float = 0.1,
        memory_H: int = 5,
        lp: int = 50,
        pi_min: float = 0.05,
        fixed_strategy: int | None = None,
        disable_credit: bool = False,
        disable_epsilon: bool | None = None,
        eps_mode: EpsilonMode = "adaptive",
        memory_scope: MemoryScope = "per_strategy",
        restart_enabled: bool = True,
        selection_mode: SelectionMode = "decomposition",
        success_criterion: SuccessCriterion = "current",
        deprecated_variant_label: str | None = None,
        adapt_fcr: bool = True,
        adapt_by_success_rate: bool = True,
        fixed_F: float = 0.5,
        fixed_CR: float = 0.5,
        delta_min: float = 0.1,
        delta_max: float = 0.9,
        stag_ratio: float = 0.1,
        restart_keep: float = 0.1,
        track_interval: int = 0,
    ) -> None:
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.seed = seed
        self._H = H
        self.T_base_ratio = T_base_ratio
        self.memory_H = memory_H
        self.lp = lp
        self.pi_min = pi_min
        self.fixed_strategy = fixed_strategy
        self.disable_credit = disable_credit
        if eps_mode == "fixed_zero":
            eps_mode = "zero"
        if disable_epsilon is True:
            if eps_mode not in {"adaptive", "zero"}:
                raise ValueError(
                    "disable_epsilon=True is deprecated and conflicts with "
                    f"eps_mode={eps_mode!r}; use eps_mode='zero' directly"
                )
            eps_mode = "zero"
        if eps_mode not in EPS_MODES:
            raise ValueError(
                f"eps_mode must be one of {sorted(EPS_MODES)}, got {eps_mode!r}"
            )
        if memory_scope not in MEMORY_SCOPES:
            raise ValueError(
                f"memory_scope must be one of {sorted(MEMORY_SCOPES)}, got {memory_scope!r}"
            )
        if selection_mode not in SELECTION_MODES:
            raise ValueError(
                f"selection_mode must be one of {sorted(SELECTION_MODES)}, "
                f"got {selection_mode!r}"
            )
        if success_criterion not in SUCCESS_CRITERIA:
            raise ValueError(
                "success_criterion must be one of "
                f"{sorted(SUCCESS_CRITERIA)}, got {success_criterion!r}"
            )
        self.eps_mode = eps_mode
        self.disable_epsilon = self.eps_mode == "zero"
        self.memory_scope = memory_scope
        self.restart_enabled = restart_enabled
        self.selection_mode = selection_mode
        self.success_criterion = success_criterion
        self.deprecated_variant_label = deprecated_variant_label
        self.adapt_fcr = bool(adapt_fcr)
        self.adapt_by_success_rate = bool(adapt_by_success_rate)
        self.fixed_F = float(fixed_F)
        self.fixed_CR = float(fixed_CR)
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.stag_ratio = stag_ratio
        self.restart_keep = restart_keep
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
        """Execute the MOSADE algorithm on the given problem.

        Parameters
        ----------
        problem : Problem
            The optimisation problem to solve.
        pf : ndarray or None
            True Pareto front, shape (P, M).  When provided and
            ``track_interval > 0``, IGD is computed at each snapshot.
            When ``None``, only HV is recorded.
        ref_point : ndarray or None
            Shared hypervolume reference point used for convergence tracking.
            When ``None``, a fallback based on the current run state is used.
        """
        rng = get_rng(self.seed)
        # FIX(audit B3): N is kept as a local variable only; self.pop_size is
        # never overwritten so the object stays stateless between run() calls.
        N = self.pop_size
        D = problem.n_var
        M = problem.n_obj

        problem.reset_eval_counter()

        # --- Weight vectors and neighborhoods ---
        H = self._H if self._H is not None else auto_partitions(N, M)
        weights = das_dennis(H, M)
        # Adjust population size to match number of weight vectors.
        N = weights.shape[0]
        log.info("Weight vectors: %d (H=%d), effective pop_size=%d", N, H, N)

        T_base = max(2, int(self.T_base_ratio * N))
        T_min = max(2, int(0.05 * N))
        T_max = max(3, int(0.30 * N))
        T_current = T_base
        neighbors = compute_neighbors(weights, T_current)

        max_gen = self.max_evals // N
        # Stagnation patience: at least 20 generations to avoid premature restarts.
        L_stag = max(20, int(self.stag_ratio * max_gen))

        # --- Initialise population ---
        X = rng.uniform(problem.lower, problem.upper, size=(N, D))
        F, CV = problem.evaluate(X)

        z_ideal = F.min(axis=0).copy()
        z_nadir = F.max(axis=0).copy()
        assoc = associate_to_weights(_normalise(F, z_ideal, z_nadir), weights)

        # --- Archive ---
        # Archive admits only CV <= 0 solutions (strictly feasible).
        # For unconstrained problems CV is always 0, so every non-dominated
        # offspring is eligible.  For constrained problems the epsilon-constraint
        # relaxation affects selection only; the archive always holds the best
        # strictly feasible non-dominated solutions found so far.
        archive = Archive(max_size=2 * N)
        archive.update(X, F, CV)

        # --- Bundle mutable state into _GenState ---
        # All helper methods receive this single object instead of individual
        # arrays, eliminating parameter ambiguity between indices and vectors.
        state = _GenState(
            X=X, F=F, CV=CV,
            assoc=assoc,
            z_ideal=z_ideal, z_nadir=z_nadir,
            weights=weights, neighbors=neighbors,
            archive=archive,
        )

        # --- LSHADE memories ---
        if self.memory_scope == "shared":
            shared_memory = LSHADEMemory(
                H=self.memory_H,
                init_F=self.fixed_F,
                init_CR=self.fixed_CR,
            )
            memories = [shared_memory for _ in range(NUM_STRATEGIES)]
        else:
            memories = [
                LSHADEMemory(H=self.memory_H, init_F=self.fixed_F, init_CR=self.fixed_CR)
                for _ in range(NUM_STRATEGIES)
            ]

        # --- Strategy selector ---
        selector = StrategySelector(window_size=self.lp, pi_min=self.pi_min)

        # --- Epsilon-constraint handler ---
        eps_handler = EpsilonConstraint()
        eps_handler.initialise(CV, max_gen)
        log.info(
            "Epsilon mode: %s (epsilon_0=%.6g, T_c=%d)",
            self.eps_mode,
            eps_handler.epsilon_0,
            eps_handler.T_c,
        )

        # --- Tracking ---
        stag_counter = 0
        restart_count = 0
        best_scal = np.inf
        # Per-subproblem scalarising values are tracked against the slot weight
        # vector j itself.  This keeps the adaptive neighbourhood logic aligned
        # with the decomposition skeleton: slot j corresponds to subproblem j.
        prev_g = np.array([
            float(tchebycheff(state.F[j], state.weights[j], state.z_ideal))
            for j in range(N)
        ])
        history: dict[str, Any] = {
            "gen": [],
            "n_evals": [],
            "strategy_probs": [],
            "strategy_use_counts": [],
            "strategy_success_counts": [],
            "strategy_credit_total": [],
            "strategy_credit_dg": [],
            "strategy_credit_dom": [],
            "memory_F_mean": [],
            "memory_CR_mean": [],
            "best_scal": [],
            "median_scal": [],
            "delta": [],
            "p_top": [],
            "T": [],
            "div_ratio": [],
            "archive_size": [],
            "restart": [],
            "restart_count": [],
            "restart_events": [],
            "epsilon": [],
            "epsilon_mode": self.eps_mode,
            "memory_scope": self.memory_scope,
            "memory_num_pools": 1 if self.memory_scope == "shared" else NUM_STRATEGIES,
            "memory_success_count_by_strategy": [],
            "memory_sampling_count_by_strategy": [],
            "selection_mode": self.selection_mode,
            "success_criterion": self.success_criterion,
            "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
            "n_feasible_before_selection": [],
            "n_feasible_after_selection": [],
            "population_nd_ratio": [],
            "feasibility_ratio": [],
            "mean_cv": [],
            "median_cv": [],
            "best_cv": [],
            "n_fronts": [],
            "truncation_method": [],
            "selection_tie_break_count": [],
            "sampled_F_mean": [],
            "sampled_F_std": [],
            "sampled_CR_mean": [],
            "sampled_CR_std": [],
            "strategy_success_rates": [],
            "success_rate_total": [],
            "success_count_total": [],
            "use_count_total": [],
            "n_constr": int(problem.n_constr),
            "adapt_fcr": bool(self.adapt_fcr),
            "adapt_by_success_rate": bool(self.adapt_by_success_rate),
            "fixed_F": float(self.fixed_F),
            "fixed_CR": float(self.fixed_CR),
        }
        if self.track_interval > 0:
            history["convergence"] = []
        _last_track_evals: int = 0  # evals at which the last convergence snapshot was taken

        gen = 0
        last_epsilon: float | None = None
        while problem.n_evals < self.max_evals:
            gen += 1
            epsilon = self._epsilon_for_generation(eps_handler, gen)
            self._validate_epsilon_value(epsilon, eps_handler, last_epsilon)
            last_epsilon = epsilon

            # Adaptive delta: probability of using global pool (local otherwise).
            progress = min(gen / max(max_gen, 1), 1.0)
            delta = self.delta_min + (self.delta_max - self.delta_min) * progress

            # Adaptive p for pbest fraction (decays 0.25 → ~0.05 over the run).
            p_top = max(2.0 / N, 0.25 - 0.20 * progress)

            # ---------------------------------------------------------------
            # Generate offspring
            # Pre-compute per-individual strategy assignments, F/CR parameters,
            # and pool-choice flags outside the loop to reduce Python overhead.
            # The per-individual loop is retained because each strategy requires
            # different parent-selection logic (branching on strat_k).
            # ---------------------------------------------------------------

            # Step 1: assign a strategy to every individual.
            if self.fixed_strategy is not None:
                # Ablation: use only a single strategy for all offspring.
                strat_assignments = np.full(N, self.fixed_strategy, dtype=int)
            else:
                # Normal: roulette-wheel on current probabilities (N calls).
                strat_assignments = np.array([selector.select(rng) for _ in range(N)])
            strat_use_counts = np.bincount(
                strat_assignments, minlength=NUM_STRATEGIES
            ).astype(int)

            # Step 2: sample F and CR grouped by strategy.  All individuals assigned
            # the same strategy are sampled together in one vectorised batch call,
            # reducing the total number of Python-level RNG dispatches from 3*N to
            # 3*K (K = number of distinct strategies active this generation).
            F_params = np.empty(N)
            CR_params = np.empty(N)
            if self.adapt_fcr:
                for _k in range(NUM_STRATEGIES):
                    _idx_k = np.where(strat_assignments == _k)[0]
                    if _idx_k.size:
                        _Fk, _CRk = memories[_k].sample_batch(_idx_k.size, rng)
                        F_params[_idx_k] = _Fk
                        CR_params[_idx_k] = _CRk
            else:
                F_params.fill(self.fixed_F)
                CR_params.fill(self.fixed_CR)

            # Step 3: pre-compute global/local pool choice (one batch of N draws).
            use_global = rng.random(N) < delta

            U = np.empty_like(state.X)
            meta: list[dict[str, Any]] = []

            for j in range(N):
                strat_k = int(strat_assignments[j])
                F_j = float(F_params[j])
                CR_j = float(CR_params[j])

                # Choose parent pool using pre-computed flag.
                if use_global[j]:
                    pool_idx = np.arange(N)
                else:
                    # Subproblem j should use neighborhood N(j), not the
                    # nearest weight-vector of the current incumbent.
                    pool_idx = np.append(state.neighbors[j], j)

                # --- Generate mutant ---
                # Verification: _pick_pbest and _pick_random_with_archive always
                # return np.ndarray decision vectors, never integer indices.
                # The mutation calls below pass those vectors directly as operands.

                if strat_k == Strategy.CURRENT_TO_PBEST_1:
                    # x_pbest: decision vector (np.ndarray, shape (D,))
                    x_pbest = self._pick_pbest(j, pool_idx, p_top, state, rng)
                    r1 = self._pick_random(pool_idx, rng, exclude={j}, n_pop=N)
                    # r2_vec: decision vector (np.ndarray, shape (D,))
                    r2_vec = self._pick_random_with_archive(state, rng, exclude={j, r1})
                    # v = x_i + F*(x_pbest - x_i) + F*(x_r1 - x_r2)
                    # Operands: all decision vectors ✓
                    v = mutate_current_to_pbest_1(
                        state.X[j], x_pbest, state.X[r1], r2_vec, F_j
                    )
                    u = binomial_crossover(state.X[j], v, CR_j, rng)

                elif strat_k == Strategy.RAND_1:
                    r1 = self._pick_random(pool_idx, rng, exclude={j}, n_pop=N)
                    r2 = self._pick_random(pool_idx, rng, exclude={j, r1}, n_pop=N)
                    r3 = self._pick_random(pool_idx, rng, exclude={j, r1, r2}, n_pop=N)
                    # v = x_r1 + F*(x_r2 - x_r3); all population indices → vectors ✓
                    v = mutate_rand_1(state.X[r1], state.X[r2], state.X[r3], F_j)
                    u = binomial_crossover(state.X[j], v, CR_j, rng)

                elif strat_k == Strategy.CURRENT_TO_RAND_1:
                    r1 = self._pick_random(pool_idx, rng, exclude={j}, n_pop=N)
                    r2 = self._pick_random(pool_idx, rng, exclude={j, r1}, n_pop=N)
                    r3 = self._pick_random(pool_idx, rng, exclude={j, r1, r2}, n_pop=N)
                    # v = x_i + K*(x_r1 - x_i) + F*(x_r2 - x_r3); no crossover ✓
                    u = mutate_current_to_rand_1(
                        state.X[j], state.X[r1], state.X[r2], state.X[r3], F_j, rng
                    )

                elif strat_k == Strategy.RAND_TO_PBEST_2:
                    # x_pbest: decision vector (np.ndarray, shape (D,))
                    x_pbest = self._pick_pbest(j, pool_idx, p_top, state, rng)
                    r1 = self._pick_random(pool_idx, rng, exclude={j}, n_pop=N)
                    # r2_vec: decision vector (np.ndarray, shape (D,))
                    r2_vec = self._pick_random_with_archive(state, rng, exclude={j, r1})
                    r3_idx = self._pick_random(pool_idx, rng, exclude={j, r1}, n_pop=N)
                    # v = X[r1] + F*(x_pbest - X[r1]) + F*(r2_vec - X[r3_idx])
                    # Operands verified:
                    #   x_r1   = state.X[r1]    — population decision vector  ✓
                    #   x_pbest = x_pbest        — decision vector from _pick_pbest ✓
                    #   x_r2   = r2_vec          — decision vector from _pick_random_with_archive ✓
                    #   x_r3   = state.X[r3_idx] — population decision vector  ✓
                    v = mutate_rand_to_pbest_2(
                        state.X[r1], x_pbest, r2_vec, state.X[r3_idx], F_j
                    )
                    u = binomial_crossover(state.X[j], v, CR_j, rng)

                else:
                    raise ValueError(f"Unknown strategy index: {strat_k}")

                # Midpoint bound repair first (catches NaN/Inf and out-of-bounds),
                # then polynomial mutation.
                u = midpoint_repair(u, state.X[j], problem.lower, problem.upper)
                u = polynomial_mutation(u, problem.lower, problem.upper, rng)

                U[j] = u
                meta.append({"strategy": strat_k, "F": F_j, "CR": CR_j, "parent": j})

            # ---------------------------------------------------------------
            # Evaluate offspring
            # FIX(audit B7): removed dead pass-block; termination is handled
            # by the while-loop condition.
            # ---------------------------------------------------------------
            F_U, CV_U = problem.evaluate(U)

            # ---------------------------------------------------------------
            # Update ideal and nadir
            # FIX(audit B2): nadir tracked as running maximum so it never
            # decreases.  The old code reset it from the current pool each
            # generation, allowing shrinkage that destabilised normalisation.
            # ---------------------------------------------------------------
            state.z_ideal, state.z_nadir = _update_running_extrema(
                state.z_ideal,
                state.z_nadir,
                F_U,
            )

            # ---------------------------------------------------------------
            # Environmental selection
            # The merged pool places parents at indices 0..N-1 and offspring
            # at indices N..2N-1.  select_per_subproblem returns one index
            # per weight vector into this 2N-element pool.
            # ---------------------------------------------------------------
            F_all = np.vstack([state.F, F_U])       # (2N, M)
            CV_all = np.concatenate([state.CV, CV_U])  # (2N,)
            X_all = np.vstack([state.X, U])          # (2N, D)

            F_norm_all = _normalise(F_all, state.z_ideal, state.z_nadir)
            assoc_all = associate_to_weights(F_norm_all, state.weights)

            n_feasible_before_selection = int(np.sum(CV_all <= 0.0))
            if self.selection_mode == "decomposition":
                sel_idx = select_per_subproblem(
                    F_all, CV_all, assoc_all, state.weights,
                    state.z_ideal, epsilon, N, state.neighbors,
                )
                selection_meta: dict[str, int | str] = {
                    "n_fronts": 0,
                    "truncation_method": "per_subproblem_tchebycheff",
                    "selection_tie_break_count": 0,
                }
            else:
                sel_idx, selection_meta = select_dominance_survival(F_all, CV_all, N)

            # ---------------------------------------------------------------
            # Credit assignment
            # sel_idx[j] is an index into the 2N pool.
            # Verification: chosen >= N correctly identifies offspring because
            # the pool is built as vstack([X, U]), so parents occupy 0..N-1
            # and offspring occupy N..2N-1.
            # ---------------------------------------------------------------
            credits = np.zeros(NUM_STRATEGIES)
            credits_dg = np.zeros(NUM_STRATEGIES)
            credits_dom = np.zeros(NUM_STRATEGIES)
            success_params: list[list[tuple[float, float, float]]] = [
                [] for _ in range(NUM_STRATEGIES)
            ]
            success_counts = np.zeros(NUM_STRATEGIES, dtype=int)
            rank_crowding_ranks: np.ndarray | None = None
            rank_crowding_cd: np.ndarray | None = None
            if self.success_criterion == "rank_crowding":
                rank_crowding_ranks = constrained_nondominated_sort(F_all, CV_all)
                rank_crowding_cd = crowding_distance_by_front(F_all, rank_crowding_ranks)

            for j in range(N):
                chosen = sel_idx[j]
                if chosen < 0:
                    continue
                if chosen >= N:
                    # An offspring was selected for subproblem j.
                    offspring_idx = chosen - N   # maps back into U / meta / F_U
                    m = meta[offspring_idx]
                    strat = m["strategy"]
                    # parent_j == offspring_idx because meta is built in the same
                    # loop order (meta[j]["parent"] == j).
                    parent_j = m["parent"]

                    credit, delta_g, dom_bonus = self._success_credit(
                        criterion=self.success_criterion,
                        parent_F=state.F[parent_j],
                        parent_CV=state.CV[parent_j],
                        child_F=F_U[offspring_idx],
                        child_CV=CV_U[offspring_idx],
                        weight=state.weights[j],
                        z_ideal=state.z_ideal,
                        F_all=F_all,
                        CV_all=CV_all,
                        parent_idx=parent_j,
                        child_idx=N + offspring_idx,
                        ranks=rank_crowding_ranks,
                        crowding=rank_crowding_cd,
                    )

                    credits[strat] += credit
                    credits_dg[strat] += delta_g
                    credits_dom[strat] += dom_bonus
                    if credit > 0:
                        success_counts[strat] += 1
                        success_params[strat].append((m["F"], m["CR"], credit))

            # ---------------------------------------------------------------
            # Update population in-place from selected indices.
            # ---------------------------------------------------------------
            valid = sel_idx >= 0
            for j in range(N):
                if valid[j]:
                    state.X[j] = X_all[sel_idx[j]]
                    state.F[j] = F_all[sel_idx[j]]
                    state.CV[j] = CV_all[sel_idx[j]]

            state.assoc = associate_to_weights(
                _normalise(state.F, state.z_ideal, state.z_nadir), state.weights
            )
            n_feasible_after_selection = int(np.sum(state.CV <= 0.0))

            # ---------------------------------------------------------------
            # Update LSHADE memories and strategy probabilities.
            # ---------------------------------------------------------------
            if self.adapt_fcr:
                if self.memory_scope == "shared":
                    pooled = [item for strat_items in success_params for item in strat_items]
                    if pooled:
                        memories[0].update(
                            [t[0] for t in pooled],
                            [t[1] for t in pooled],
                            [t[2] for t in pooled],
                        )
                else:
                    for k in range(NUM_STRATEGIES):
                        if success_params[k]:
                            sf = [t[0] for t in success_params[k]]
                            scr = [t[1] for t in success_params[k]]
                            w = [t[2] for t in success_params[k]]
                            memories[k].update(sf, scr, w)

            if (
                self.adapt_by_success_rate
                and not self.disable_credit
                and self.fixed_strategy is None
            ):
                selector.update(credits)

            # ---------------------------------------------------------------
            # Update archive with offspring.
            # Archive.update admits only CV <= 0.0 (strictly feasible).
            # For unconstrained problems CV is always 0 so all non-dominated
            # offspring are eligible.  For constrained problems the epsilon
            # relaxation used in selection is deliberately NOT applied here:
            # the archive records only solutions that are genuinely feasible.
            # ---------------------------------------------------------------
            state.archive.update(U, F_U, CV_U)

            # ---------------------------------------------------------------
            # Adaptive neighbourhood size T.
            # ---------------------------------------------------------------
            curr_g = np.array([
                float(tchebycheff(state.F[j], state.weights[j], state.z_ideal))
                for j in range(N)
            ])
            improved = np.sum(curr_g < prev_g - 1e-12)
            div_ratio = improved / N
            alpha_T = 0.5
            T_new = int(round(T_base * (1.0 + alpha_T * (div_ratio - 0.5))))
            T_new = max(T_min, min(T_max, T_new))
            if T_new != T_current:
                T_current = T_new
                state.neighbors = compute_neighbors(state.weights, T_current)
            prev_g = curr_g

            # ---------------------------------------------------------------
            # Stagnation check → restart if needed.
            # ---------------------------------------------------------------
            restart_triggered = False
            current_best_scal = float(curr_g.min())
            median_scal = float(np.median(curr_g))
            if best_scal - current_best_scal > 1e-6:
                best_scal = current_best_scal
                stag_counter = 0
            else:
                stag_counter += 1

            if self.restart_enabled and stag_counter >= L_stag:
                log.info("Stagnation restart at gen %d", gen)
                pre_restart_diversity = _mean_pairwise_decision_distance(state.X)
                self._restart(state, problem, rng, memories, selector)
                post_restart_diversity = _mean_pairwise_decision_distance(state.X)
                # Re-derive normalisation and associations from the restarted
                # population while preserving running ideal/nadir semantics.
                state.z_ideal, state.z_nadir = _update_running_extrema(
                    state.z_ideal,
                    state.z_nadir,
                    state.F,
                )
                state.assoc = associate_to_weights(
                    _normalise(state.F, state.z_ideal, state.z_nadir), state.weights
                )
                restart_count += 1
                restart_triggered = True
                stag_counter = 0
                prev_g = np.array([
                    float(tchebycheff(state.F[j], state.weights[j], state.z_ideal))
                    for j in range(N)
                ])
                current_best_scal = float(prev_g.min())
                median_scal = float(np.median(prev_g))
                best_scal = current_best_scal
                history["restart_events"].append({
                    "gen": int(gen),
                    "n_evals": int(problem.n_evals),
                    "restart_reason": "stagnation",
                    "restart_fraction": float(1.0 - self.restart_keep),
                    "n_individuals_reinitialized": int(N - max(1, int(self.restart_keep * N))),
                    "pre_restart_diversity": float(pre_restart_diversity),
                    "post_restart_diversity": float(post_restart_diversity),
                })

            # ---------------------------------------------------------------
            # Record history.
            # ---------------------------------------------------------------
            if self.fixed_strategy is not None:
                logged_probs = np.zeros(NUM_STRATEGIES, dtype=float)
                logged_probs[self.fixed_strategy] = 1.0
            else:
                logged_probs = selector.probabilities

            archive_size = (
                0
                if state.archive.is_empty
                else int(state.archive.get_objectives().shape[0])
            )
            feasible_mask = state.CV <= 0.0
            if np.any(feasible_mask):
                nd_mask = nondominated_mask(state.F[feasible_mask])
                population_nd_ratio = float(np.sum(nd_mask) / N)
            else:
                nd_mask = nondominated_mask(state.F)
                population_nd_ratio = float(np.sum(nd_mask) / N)
            strategy_success_rates = np.divide(
                success_counts,
                strat_use_counts,
                out=np.zeros(NUM_STRATEGIES, dtype=float),
                where=strat_use_counts > 0,
            )
            total_uses = int(np.sum(strat_use_counts))
            total_successes = int(np.sum(success_counts))

            history["gen"].append(gen)
            history["n_evals"].append(problem.n_evals)
            history["strategy_probs"].append(logged_probs.tolist())
            history["strategy_use_counts"].append(strat_use_counts.tolist())
            history["strategy_success_counts"].append(success_counts.tolist())
            history["strategy_credit_total"].append(credits.tolist())
            history["strategy_credit_dg"].append(credits_dg.tolist())
            history["strategy_credit_dom"].append(credits_dom.tolist())
            history["memory_F_mean"].append([mem.mean_F for mem in memories])
            history["memory_CR_mean"].append([mem.mean_CR for mem in memories])
            history["sampled_F_mean"].append(float(np.mean(F_params)))
            history["sampled_F_std"].append(float(np.std(F_params, ddof=0)))
            history["sampled_CR_mean"].append(float(np.mean(CR_params)))
            history["sampled_CR_std"].append(float(np.std(CR_params, ddof=0)))
            history["strategy_success_rates"].append(strategy_success_rates.tolist())
            history["success_rate_total"].append(
                float(total_successes / total_uses) if total_uses > 0 else 0.0
            )
            history["success_count_total"].append(total_successes)
            history["use_count_total"].append(total_uses)
            history["best_scal"].append(current_best_scal)
            history["median_scal"].append(median_scal)
            history["delta"].append(float(delta))
            history["p_top"].append(float(p_top))
            history["T"].append(int(T_current))
            history["div_ratio"].append(float(div_ratio))
            history["archive_size"].append(archive_size)
            history["restart"].append(bool(restart_triggered))
            history["restart_count"].append(int(restart_count))
            history["epsilon"].append(float(epsilon))
            history["memory_success_count_by_strategy"].append(success_counts.tolist())
            history["memory_sampling_count_by_strategy"].append(strat_use_counts.tolist())
            history["n_feasible_before_selection"].append(n_feasible_before_selection)
            history["n_feasible_after_selection"].append(n_feasible_after_selection)
            history["population_nd_ratio"].append(population_nd_ratio)
            history["feasibility_ratio"].append(float(np.mean(state.CV <= 0.0)))
            history["mean_cv"].append(float(np.mean(state.CV)))
            history["median_cv"].append(float(np.median(state.CV)))
            history["best_cv"].append(float(np.min(state.CV)))
            history["n_fronts"].append(int(selection_meta.get("n_fronts", 0)))
            history["truncation_method"].append(str(selection_meta.get("truncation_method", "")))
            history["selection_tie_break_count"].append(
                int(selection_meta.get("selection_tie_break_count", 0))
            )

            # ---------------------------------------------------------------
            # Convergence tracking (optional).
            # ---------------------------------------------------------------
            if self.track_interval > 0 and (
                problem.n_evals - _last_track_evals >= self.track_interval
            ):
                snap = self._convergence_snapshot(state, pf, ref_point, problem.n_evals)
                history["convergence"].append(snap)
                _last_track_evals = problem.n_evals

        # --- Prepare output ---
        if state.archive.is_empty:
            out_X, out_F = state.X.copy(), state.F.copy()
        else:
            out_X = state.archive.get_decisions()
            out_F = state.archive.get_objectives()

        epsilon_history = [float(v) for v in history.get("epsilon", [])]
        epsilon_history_json = json.dumps(
            epsilon_history,
            separators=(",", ":"),
            sort_keys=True,
        )
        epsilon_history_sha256 = hashlib.sha256(
            epsilon_history_json.encode("utf-8")
        ).hexdigest()
        memory_digest_payload = {
            "scope": self.memory_scope,
            "F": history.get("memory_F_mean", []),
            "CR": history.get("memory_CR_mean", []),
        }
        memory_history_digest = hashlib.sha256(
            json.dumps(
                memory_digest_payload,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        return MOSADEResult(
            X=out_X,
            F=out_F,
            n_evals=problem.n_evals,
            history=history,
            metadata={
                "pf_source": (
                    "archive"
                    if not state.archive.is_empty
                    else "final_population_feasible_nondominated"
                ),
                "seed": self.seed,
                "eps_mode": self.eps_mode,
                "adaptive_eps": self.eps_mode == "adaptive",
                "disable_epsilon": self.disable_epsilon,
                "epsilon_initial": float(eps_handler.epsilon_0),
                "epsilon_0": float(eps_handler.epsilon_0),
                "epsilon_T_c": int(eps_handler.T_c),
                "epsilon_final": float(last_epsilon if last_epsilon is not None else 0.0),
                "epsilon_min": float(np.min(epsilon_history)) if epsilon_history else 0.0,
                "epsilon_max": float(np.max(epsilon_history)) if epsilon_history else 0.0,
                "epsilon_num_updates": int(
                    np.sum(np.abs(np.diff(epsilon_history)) > 1e-12)
                ) if len(epsilon_history) > 1 else 0,
                "epsilon_history_sha256": epsilon_history_sha256,
                "constraint_handling_mode": "epsilon_constraint",
                "memory_scope": self.memory_scope,
                "memory_num_pools": 1 if self.memory_scope == "shared" else NUM_STRATEGIES,
                "memory_update_count": int(sum(sum(v) for v in history["memory_success_count_by_strategy"])),
                "memory_success_count_by_strategy": np.sum(
                    np.asarray(history["memory_success_count_by_strategy"], dtype=int),
                    axis=0,
                ).astype(int).tolist() if history["memory_success_count_by_strategy"] else [0] * NUM_STRATEGIES,
                "memory_sampling_count_by_strategy": np.sum(
                    np.asarray(history["memory_sampling_count_by_strategy"], dtype=int),
                    axis=0,
                ).astype(int).tolist() if history["memory_sampling_count_by_strategy"] else [0] * NUM_STRATEGIES,
                "memory_history_digest": memory_history_digest,
                "restart_enabled": bool(self.restart_enabled),
                "restart_count": int(restart_count),
                "restart_generation_or_eval": history["restart_events"],
                "selection_mode": self.selection_mode,
                "success_criterion": self.success_criterion,
                "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
                "deprecated_variant_label": self.deprecated_variant_label,
                "adapt_fcr": bool(self.adapt_fcr),
                "adapt_by_success_rate": bool(self.adapt_by_success_rate),
                "disable_credit": bool(self.disable_credit),
                "fixed_F": float(self.fixed_F),
                "fixed_CR": float(self.fixed_CR),
                "effective_pop_size": int(N),
                "n_weights": int(N),
                "n_obj": int(M),
                "n_var": int(D),
            },
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _epsilon_for_generation(
        self,
        eps_handler: EpsilonConstraint,
        gen: int,
    ) -> float:
        """Return the configured epsilon value for generation *gen*."""
        if self.eps_mode == "adaptive":
            return float(eps_handler(gen))
        if self.eps_mode == "fixed_initial":
            return float(eps_handler.epsilon_0)
        if self.eps_mode == "zero":
            return 0.0
        raise ValueError(f"Unknown eps_mode: {self.eps_mode!r}")

    def _validate_epsilon_value(
        self,
        epsilon: float,
        eps_handler: EpsilonConstraint,
        last_epsilon: float | None,
    ) -> None:
        """Fail fast if a configured epsilon mode violates its definition."""
        if self.eps_mode == "adaptive":
            if last_epsilon is not None and epsilon > last_epsilon + 1e-12:
                raise RuntimeError("adaptive epsilon increased between generations")
            return
        if self.eps_mode == "fixed_initial":
            if not np.isclose(epsilon, eps_handler.epsilon_0, rtol=0.0, atol=1e-12):
                raise RuntimeError("fixed_initial epsilon changed during the run")
            return
        if self.eps_mode == "zero" and abs(epsilon) > 1e-12:
            raise RuntimeError("zero epsilon mode is not zero")

    @staticmethod
    def _success_credit(
        criterion: str,
        parent_F: np.ndarray,
        parent_CV: float,
        child_F: np.ndarray,
        child_CV: float,
        weight: np.ndarray,
        z_ideal: np.ndarray,
        F_all: np.ndarray,
        CV_all: np.ndarray,
        parent_idx: int,
        child_idx: int,
        ranks: np.ndarray | None = None,
        crowding: np.ndarray | None = None,
    ) -> tuple[float, float, float]:
        """Return diagnostic adaptation credit and component telemetry.

        The default ``current`` criterion is the original ScienceN MOSADE
        success signal: Tchebycheff improvement plus a Pareto-dominance bonus.
        Other criteria are diagnostic-only alternatives used to test whether
        the success signal is misaligned with multi-objective progress.
        """
        g_parent = float(tchebycheff(parent_F, weight, z_ideal))
        g_child = float(tchebycheff(child_F, weight, z_ideal))
        delta_g = max(0.0, g_parent - g_child)
        dom_bonus = 0.5 if dominates(child_F, parent_F) else 0.0

        if criterion == "current":
            return delta_g + dom_bonus, delta_g, dom_bonus

        parent_feasible = parent_CV <= 0.0
        child_feasible = child_CV <= 0.0

        if criterion == "decomposition":
            return delta_g, delta_g, 0.0

        if criterion == "dominance":
            if child_feasible and not parent_feasible:
                credit = 1.0 + max(0.0, float(parent_CV - child_CV))
            elif not child_feasible and not parent_feasible:
                credit = max(0.0, float(parent_CV - child_CV))
            elif child_feasible and parent_feasible and dominates(child_F, parent_F):
                credit = 1.0
            else:
                credit = 0.0
            return credit, credit, 1.0 if credit > 0.0 else 0.0

        if criterion == "feasibility_first":
            cv_delta = max(0.0, float(parent_CV - child_CV))
            if child_feasible and not parent_feasible:
                credit = 1.0 + cv_delta
            elif not child_feasible and not parent_feasible:
                credit = cv_delta
            else:
                credit = delta_g + dom_bonus
            return credit, delta_g, cv_delta

        if criterion == "rank_crowding":
            if ranks is None or crowding is None:
                return delta_g + dom_bonus, delta_g, dom_bonus
            parent_rank = int(ranks[parent_idx])
            child_rank = int(ranks[child_idx])
            parent_cd = float(crowding[parent_idx])
            child_cd = float(crowding[child_idx])
            if child_rank < parent_rank:
                credit = float(parent_rank - child_rank + 1)
            elif child_rank == parent_rank and child_cd > parent_cd:
                if np.isinf(child_cd) and not np.isinf(parent_cd):
                    credit = 1.0
                else:
                    credit = max(1e-6, child_cd - parent_cd)
            else:
                credit = 0.0
            return credit, delta_g, 1.0 if credit > 0.0 else 0.0

        if criterion == "hvproxy":
            # Cheap minimisation proxy: reward objective-wise movement toward
            # the ideal point, with feasibility handled before objective gains.
            cv_delta = max(0.0, float(parent_CV - child_CV))
            if child_feasible and not parent_feasible:
                credit = 1.0 + cv_delta
            elif not child_feasible and not parent_feasible:
                credit = cv_delta
            else:
                scale = np.maximum(np.abs(z_ideal), 1.0)
                improvement = np.maximum(0.0, (parent_F - child_F) / scale)
                credit = float(np.sum(improvement))
            return credit, delta_g, cv_delta

        raise ValueError(f"Unknown success_criterion: {criterion!r}")

    @staticmethod
    def _pick_random(
        pool: np.ndarray,
        rng: np.random.Generator,
        exclude: set[int],
        n_pop: int,
    ) -> int:
        """Return a random population *index* from *pool* excluding given indices.

        Returns an ``int``, never a decision vector.

        FIX(audit B3): accepts ``n_pop`` explicitly so this method is correct
        even when called on different problems with different effective N.
        FIX(audit B5): the global-population fallback is intentional and logged
        at DEBUG level so it can be traced without polluting normal output.
        """
        candidates = [i for i in pool if i not in exclude]
        if not candidates:
            log.debug(
                "_pick_random: pool exhausted after exclusions; "
                "falling back to global population (n_pop=%d)", n_pop
            )
            candidates = [i for i in range(n_pop) if i not in exclude]
        return int(rng.choice(candidates))

    @staticmethod
    def _pick_random_with_archive(
        state: _GenState,
        rng: np.random.Generator,
        exclude: set[int],
    ) -> np.ndarray:
        """Return a decision vector sampled from population ∪ archive.

        Always returns an ``np.ndarray`` of shape ``(D,)`` — never an index.
        The ``exclude`` set contains **population** indices to skip; archive
        members are never excluded (they use separate internal indices).

        Used as the ``x_r2`` operand in strategies S1 and S4.
        """
        if not state.archive.is_empty and rng.random() < 0.5:
            # archive.random_member returns a decision vector directly.
            return state.archive.random_member(rng)
        N = state.N
        candidates = [i for i in range(N) if i not in exclude]
        if not candidates:
            candidates = list(range(N))
        # Return the decision vector, not the index.
        return state.X[rng.choice(candidates)].copy()

    @staticmethod
    def _pick_pbest(
        j: int,
        pool_idx: np.ndarray,
        p_top: float,
        state: _GenState,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return the x_pbest decision vector for subproblem *j*.

        Always returns an ``np.ndarray`` of shape ``(D,)`` — never an index.

        Ranks pool candidates by Tchebycheff value under subproblem j's weight
        vector, selects the top-p fraction, then either draws uniformly from
        that fraction (70% chance) or draws from the archive's top-p fraction
        (30% chance).

        Used as the ``x_pbest`` operand in strategies S1 and S4.
        """
        w = state.weights[j]

        # Rank pool by scalarising value under this subproblem's weight.
        g_vals = np.array([
            float(tchebycheff(state.F[i], w, state.z_ideal)) for i in pool_idx
        ])
        n_top = max(1, int(p_top * len(pool_idx)))
        top_idx = pool_idx[np.argsort(g_vals)[:n_top]]

        # With 30% probability draw from archive top instead of population top.
        if not state.archive.is_empty and rng.random() < 0.3:
            arc_F = state.archive.get_objectives()
            arc_X = state.archive.get_decisions()
            arc_g = np.array([
                float(tchebycheff(arc_F[i], w, state.z_ideal))
                for i in range(arc_F.shape[0])
            ])
            arc_top = np.argsort(arc_g)[:max(1, int(p_top * len(arc_g)))]
            # Return decision vector from archive — never an index.
            return arc_X[rng.choice(arc_top)].copy()

        # Return decision vector from population — never an index.
        return state.X[top_idx[rng.integers(len(top_idx))]].copy()

    def _restart(
        self,
        state: _GenState,
        problem: Problem,
        rng: np.random.Generator,
        memories: list[LSHADEMemory],
        selector: StrategySelector,
    ) -> None:
        """Stagnation restart: keep top restart_keep fraction, re-initialise rest.

        Mutates ``state.X``, ``state.F``, ``state.CV`` in-place.
        The caller is responsible for updating ``state.z_ideal``,
        ``state.z_nadir``, and ``state.assoc`` afterwards.

        FIX(audit B6): argsort called once via ``order``; the original code
        called it twice and left ``keep_idx`` as unused dead code.
        """
        N, D = state.X.shape
        n_keep = max(1, int(self.restart_keep * N))

        # Rank by uniform-weight scalarisation as a rough quality proxy.
        w_uniform = np.ones(state.F.shape[1]) / state.F.shape[1]
        z_ideal_local = state.F.min(axis=0)
        g_vals = np.array([
            float(tchebycheff(state.F[i], w_uniform, z_ideal_local)) for i in range(N)
        ])
        order = np.argsort(g_vals)       # best → worst
        replace_idx = order[n_keep:]     # indices to reinitialise

        state.X[replace_idx] = rng.uniform(
            problem.lower, problem.upper, size=(len(replace_idx), D)
        )
        F_new, CV_new = problem.evaluate(state.X[replace_idx])
        state.F[replace_idx] = F_new
        state.CV[replace_idx] = CV_new

        seen_memories: set[int] = set()
        for mem in memories:
            if id(mem) in seen_memories:
                continue
            seen_memories.add(id(mem))
            mem.reset()
        selector.reset()

    @staticmethod
    def _convergence_snapshot(
        state: _GenState,
        pf: np.ndarray | None,
        ref_point: np.ndarray | None,
        n_evals: int,
    ) -> dict[str, object]:
        """Compute a convergence snapshot from the current population.

        Uses the nondominated feasible subset of the population (not the
        archive) so that the metric reflects the live search state rather
        than the best-seen accumulation.

        Parameters
        ----------
        state : _GenState
            Current generation state.
        pf : ndarray or None
            True Pareto front for IGD computation.  When ``None`` only HV
            is computed.
        n_evals : int
            Current evaluation count (from ``problem.n_evals``).

        Returns
        -------
        dict with keys ``n_evals`` (int), ``hv`` (float), ``igd`` (float or None).
        """
        # Feasibility mask: CV == 0 (strictly feasible).
        feasible_mask = state.CV <= 0.0
        if feasible_mask.any():
            F_feas = state.F[feasible_mask]
            nd_mask = nondominated_mask(F_feas)
            F_nd = F_feas[nd_mask]
        else:
            # No feasible solutions yet — use all nondominated (for constrained runs).
            F_nd = np.empty((0, state.F.shape[1]), dtype=float)

        ref = (
            ref_point
            if ref_point is not None
            else _safe_reference_point_from_maxima(state.z_nadir)
        )

        hv_val = hypervolume(F_nd, ref) if F_nd.shape[0] > 0 else 0.0

        igd_val: float | None = None
        if pf is not None and F_nd.shape[0] > 0:
            igd_val = compute_igd(F_nd, pf)

        return {"n_evals": n_evals, "hv": hv_val, "igd": igd_val}


# ---------------------------------------------------------------------------
# Module-level utility
# ---------------------------------------------------------------------------

def _safe_reference_point_from_maxima(maxima: np.ndarray) -> np.ndarray:
    """Return a minimisation HV reference point that safely dominates *maxima*.

    ``maxima * 1.1`` fails when maxima are negative because it moves the
    reference point in the wrong direction.  We instead add a positive margin
    proportional to the absolute scale of each objective.
    """
    maxima = np.asarray(maxima, dtype=float)
    margin = np.maximum(1e-6, 0.1 * np.maximum(np.abs(maxima), 1.0))
    return maxima + margin


def _normalise(F: np.ndarray, z_ideal: np.ndarray, z_nadir: np.ndarray) -> np.ndarray:
    """Normalise objectives to [0, ~1] using ideal and nadir points."""
    denom = z_nadir - z_ideal
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return (F - z_ideal) / denom


def _mean_pairwise_decision_distance(X: np.ndarray) -> float:
    """Return a compact diversity proxy from pairwise decision distances."""
    if X.shape[0] < 2:
        return 0.0
    diffs = X[:, None, :] - X[None, :, :]
    dist = np.sqrt(np.sum(diffs * diffs, axis=2))
    tri = dist[np.triu_indices(X.shape[0], k=1)]
    return float(np.mean(tri)) if tri.size else 0.0


def _update_running_extrema(
    z_ideal: np.ndarray,
    z_nadir: np.ndarray,
    F: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Update running ideal/nadir points without allowing regressions.

    Parameters
    ----------
    z_ideal, z_nadir : ndarray, shape (M,)
        Previously seen ideal/nadir extrema.
    F : ndarray, shape (N, M)
        New objective vectors to fold into the running extrema.

    Returns
    -------
    tuple(ndarray, ndarray)
        Updated ``(z_ideal, z_nadir)`` where the ideal never increases and
        the nadir never decreases.
    """
    return np.minimum(z_ideal, F.min(axis=0)), np.maximum(z_nadir, F.max(axis=0))
