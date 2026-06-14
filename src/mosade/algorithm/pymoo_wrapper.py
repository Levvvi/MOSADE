"""Wrappers for pymoo algorithms, adapting them to the MOSADE experiment interface.

Provides ``PymooAlgorithm``, a thin adapter that:

1. Converts a :class:`mosade.problems.base.Problem` to a pymoo ``ElementwiseProblem``
   (or ``Problem``) so pymoo can call it.
2. Constructs the requested pymoo algorithm (NSGA3, SPEA2, SMS-EMOA, MOEA/D-DE).
3. Runs ``pymoo.optimize.minimize`` and returns a :class:`MOSADEResult`.

Supported algorithm names
--------------------------
* ``"NSGA3"``    — NSGA-III (reference-direction based)
* ``"SPEA2"``    — SPEA2
* ``"SMSEMOA"``  — SMS-EMOA
* ``"MOEAD_DE"`` — MOEA/D with DE/rand/1 crossover

All algorithms are imported lazily inside :meth:`PymooAlgorithm.run` so that
the module can be imported without pymoo installed.

Usage (via experiment YAML)::

    algorithms:
      - name: NSGA3
        type: NSGA3
        pop_size: 100
        max_evals: 50000
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

from mosade.algorithm.mosade import MOSADEResult

log = logging.getLogger("mosade.runner")


class PymooAlgorithm:
    """Adapter wrapping a pymoo multi-objective algorithm.

    Parameters
    ----------
    algo_name : str
        One of ``"NSGA3"``, ``"SPEA2"``, ``"SMSEMOA"``, ``"MOEAD_DE"``.
    pop_size : int
        Target population size (may be rounded for reference-direction methods).
    max_evals : int
        Total function evaluation budget passed to pymoo as termination.
    seed : int
        Random seed forwarded to ``pymoo.optimize.minimize``.
    **algo_kwargs :
        Extra keyword arguments forwarded to the pymoo algorithm constructor.
    """

    # Keys that are MOSADE-specific and should not be forwarded to pymoo.
    _MOSADE_ONLY_KEYS: frozenset[str] = frozenset({
        "track_interval", "T_base_ratio", "memory_H", "lp", "pi_min",
        "fixed_strategy", "disable_credit", "disable_epsilon",
        "delta_min", "delta_max", "stag_ratio", "restart_keep",
    })

    def __init__(
        self,
        algo_name: str,
        pop_size: int = 100,
        max_evals: int = 100_000,
        seed: int = 42,
        **algo_kwargs: Any,
    ) -> None:
        self.algo_name = algo_name
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.seed = seed
        # Strip MOSADE-specific keys that have no meaning for pymoo algorithms
        # so that they don't leak into pymoo constructor calls.
        self.algo_kwargs = {
            k: v for k, v in algo_kwargs.items()
            if k not in self._MOSADE_ONLY_KEYS
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def unsupported_reason(self, problem: Any) -> str | None:
        """Return a human-readable reason when *problem* is unsupported."""
        if problem.n_constr > 0 and self.algo_name.upper() == "MOEAD_DE":
            return "constrained problems are not implemented for MOEAD_DE"
        return None

    def run(
        self,
        problem: Any,
        pf: np.ndarray | None = None,
        ref_point: np.ndarray | None = None,
    ) -> MOSADEResult:
        """Run the pymoo algorithm on *problem* and return a :class:`MOSADEResult`.

        Parameters
        ----------
        problem :
            A :class:`mosade.problems.base.Problem` instance.
        pf :
            Ignored (present only for interface compatibility).
        ref_point :
            Ignored by pymoo-backed algorithms. Present for interface
            compatibility with the experiment runner and convergence-aware
            algorithms.

        Returns
        -------
        MOSADEResult
        """
        reason = self.unsupported_reason(problem)
        if reason is not None:
            log.warning(
                "%s unsupported on constrained problem %s; "
                "skipping this algorithm/problem pair",
                self.algo_name,
                problem.__class__.__name__,
            )
            return MOSADEResult(
                X=np.empty((0, problem.n_var)),
                F=np.empty((0, problem.n_obj)),
                n_evals=0,
                history={},
                metadata={"pf_source": "unsupported"},
                status="unsupported",
                message=reason,
            )

        try:
            import pymoo  # noqa: F401 — presence check
        except ImportError as exc:
            raise ImportError(
                "pymoo is required to use PymooAlgorithm. "
                "Install it with: pip install pymoo"
            ) from exc

        from pymoo.core.problem import Problem as PymooProblem
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination

        # ----------------------------------------------------------------
        # Problem adapter
        # ----------------------------------------------------------------
        class _Adapter(PymooProblem):
            """Bridges a mosade Problem into the pymoo Problem interface."""

            def __init__(self_, our_problem: Any) -> None:
                n_ieq = our_problem.n_constr
                super().__init__(
                    n_var=our_problem.n_var,
                    n_obj=our_problem.n_obj,
                    n_ieq_constr=n_ieq,
                    xl=our_problem.lower.copy(),
                    xu=our_problem.upper.copy(),
                )
                self_._our_problem = our_problem
                self_._eval_count = 0

            def _evaluate(self_, X: np.ndarray, out: dict, *args: Any, **kwargs: Any) -> None:
                X = np.atleast_2d(X)
                F, G = self_._our_problem._evaluate(X)
                self_._eval_count += X.shape[0]
                out["F"] = F
                if G is not None and G.shape[1] > 0:
                    # pymoo uses g <= 0 feasible, same as our convention
                    out["G"] = G

        adapter = _Adapter(problem)
        algo = self._build_pymoo_algo(problem.n_obj)
        termination = get_termination("n_eval", self.max_evals)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    adapter,
                    algo,
                    termination,
                    seed=self.seed,
                    verbose=False,
                )
        except AssertionError as exc:
            # Some pymoo algorithms (e.g. MOEAD) explicitly reject constrained
            # problems.  Return an empty result so the runner can record N/A
            # metrics rather than crashing the entire experiment.
            if "constraint" in str(exc).lower():
                log.warning(
                    "%s does not support constrained problems — skipping run "
                    "(problem.n_constr=%d).  Metrics will be NaN for this run.",
                    self.algo_name, problem.n_constr,
                )
                return MOSADEResult(
                    X=np.empty((0, problem.n_var)),
                    F=np.empty((0, problem.n_obj)),
                    n_evals=0,
                    history={},
                    metadata={"pf_source": "unsupported"},
                    status="unsupported",
                    message=str(exc),
                )
            raise  # re-raise unrelated assertion errors

        # res.X / res.F may be None if no feasible solution was found
        X_out = res.X if res.X is not None else np.empty((0, problem.n_var))
        F_out = res.F if res.F is not None else np.empty((0, problem.n_obj))

        return MOSADEResult(
            X=np.atleast_2d(X_out),
            F=np.atleast_2d(F_out),
            n_evals=adapter._eval_count,
            history={},
            metadata={"pf_source": "final_population_feasible_nondominated"},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_ref_dirs(self, n_obj: int) -> np.ndarray:
        """Build Das-Dennis reference directions matching *self.pop_size*.

        Uses :func:`mosade.algorithm.decomposition.auto_partitions` to find the
        largest H with C(H+M-1, M-1) ≤ pop_size, then calls the pymoo
        reference-direction factory.
        """
        from pymoo.util.ref_dirs import get_reference_directions

        from mosade.algorithm.decomposition import auto_partitions

        H = auto_partitions(self.pop_size, n_obj)
        H = max(H, 1)  # at least 1 partition
        return get_reference_directions("das-dennis", n_obj, n_partitions=H)

    def _build_pymoo_algo(self, n_obj: int) -> Any:
        """Construct and return the pymoo algorithm instance."""
        name = self.algo_name.upper()

        if name == "NSGA3":
            from pymoo.algorithms.moo.nsga3 import NSGA3

            ref_dirs = self._make_ref_dirs(n_obj)
            return NSGA3(ref_dirs=ref_dirs, **self.algo_kwargs)

        if name == "SPEA2":
            from pymoo.algorithms.moo.spea2 import SPEA2

            return SPEA2(pop_size=self.pop_size, **self.algo_kwargs)

        if name == "SMSEMOA":
            from pymoo.algorithms.moo.sms import SMSEMOA

            return SMSEMOA(pop_size=self.pop_size, **self.algo_kwargs)

        if name == "MOEAD_DE":
            from pymoo.algorithms.moo.moead import MOEAD
            from pymoo.operators.crossover.dex import DEX
            from pymoo.operators.mutation.pm import PM

            ref_dirs = self._make_ref_dirs(n_obj)
            crossover = DEX(F=0.5, CR=1.0)
            mutation = PM(eta=20)
            return MOEAD(
                ref_dirs=ref_dirs,
                crossover=crossover,
                mutation=mutation,
                **self.algo_kwargs,
            )

        raise ValueError(
            f"Unknown pymoo algorithm name: {self.algo_name!r}. "
            "Supported: NSGA3, SPEA2, SMSEMOA, MOEAD_DE"
        )
