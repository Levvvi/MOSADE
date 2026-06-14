"""Self-adaptation mechanisms for MOSADE.

- Per-strategy LSHADE-style success memories for F and CR
- Credit-based sliding-window strategy selection probabilities
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from mosade.algorithm.strategies import NUM_STRATEGIES


# ---------------------------------------------------------------------------
# LSHADE success memory
# ---------------------------------------------------------------------------


class LSHADEMemory:
    """Independent LSHADE-style success memory for one strategy.

    Stores H historical mean values for F and CR, updated by weighted
    Lehmer / weighted arithmetic means of successful parameter values.
    """

    def __init__(self, H: int = 5, init_F: float = 0.5, init_CR: float = 0.5) -> None:
        self.H = H
        self.M_F = np.full(H, init_F)
        self.M_CR = np.full(H, init_CR)
        self._k = 0  # circular write index

    def sample(self, rng: np.random.Generator) -> tuple[float, float]:
        """Sample (F, CR) from the memory."""
        r = rng.integers(self.H)
        F = float(np.clip(rng.standard_cauchy() * 0.1 + self.M_F[r], 1e-6, 1.0))
        CR = float(np.clip(rng.normal(self.M_CR[r], 0.1), 0.0, 1.0))
        return F, CR

    def sample_batch(
        self, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample n (F, CR) pairs from the memory in one vectorised call.

        Parameters
        ----------
        n : int
            Number of pairs to sample.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        F_vals : ndarray, shape (n,)
        CR_vals : ndarray, shape (n,)
        """
        if n == 0:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        r = rng.integers(self.H, size=n)
        F_vals = np.clip(rng.standard_cauchy(n) * 0.1 + self.M_F[r], 1e-6, 1.0)
        CR_vals = np.clip(rng.normal(self.M_CR[r], 0.1, size=n), 0.0, 1.0)
        return F_vals, CR_vals

    def update(self, S_F: list[float], S_CR: list[float], weights: list[float]) -> None:
        """Update memory with successful F/CR pairs.

        Parameters
        ----------
        S_F, S_CR : lists of successful F and CR values
        weights : improvement-based weights (same length as S_F/S_CR)
        """
        if len(S_F) == 0:
            return
        w = np.array(weights)
        w = w / (w.sum() + 1e-30)  # normalise

        sf = np.array(S_F)
        scr = np.array(S_CR)

        # Weighted Lehmer mean for F
        mean_F = float(np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30))
        # Weighted arithmetic mean for CR
        mean_CR = float(np.sum(w * scr))

        self.M_F[self._k] = mean_F
        self.M_CR[self._k] = mean_CR
        self._k = (self._k + 1) % self.H

    def reset(self, init_F: float = 0.5, init_CR: float = 0.5) -> None:
        self.M_F[:] = init_F
        self.M_CR[:] = init_CR
        self._k = 0

    @property
    def mean_F(self) -> float:
        """Return the current mean of the F memory.

        Used by run-history telemetry so we can verify that each strategy keeps
        an independent success memory and that the memories are actually moving
        over time.
        """
        return float(np.mean(self.M_F))

    @property
    def mean_CR(self) -> float:
        """Return the current mean of the CR memory."""
        return float(np.mean(self.M_CR))

    def snapshot(self) -> dict[str, list[float] | float]:
        """Return a lightweight serialisable view of the memory state."""
        return {
            "M_F": self.M_F.tolist(),
            "M_CR": self.M_CR.tolist(),
            "mean_F": self.mean_F,
            "mean_CR": self.mean_CR,
        }


# ---------------------------------------------------------------------------
# Strategy selection via credit assignment
# ---------------------------------------------------------------------------


@dataclass
class StrategySelector:
    """Credit-based adaptive strategy selection with sliding window.

    Maintains selection probabilities π_k for each strategy, updated
    from cumulative credit over a sliding window of recent generations.
    """

    n_strategies: int = NUM_STRATEGIES
    window_size: int = 50  # LP in the design doc
    pi_min: float = 0.05

    # internal
    _pi: np.ndarray = field(init=False)
    _credit_history: deque = field(init=False)

    def __post_init__(self) -> None:
        self._pi = np.full(self.n_strategies, 1.0 / self.n_strategies)
        self._credit_history = deque(maxlen=self.window_size)

    @property
    def probabilities(self) -> np.ndarray:
        return self._pi.copy()

    @property
    def credit_totals(self) -> np.ndarray:
        """Return sliding-window cumulative credits for logging/debugging."""
        totals = np.zeros(self.n_strategies)
        for c in self._credit_history:
            totals += c
        return totals

    def select(self, rng: np.random.Generator) -> int:
        """Sample a strategy index using roulette-wheel on current probabilities."""
        return int(rng.choice(self.n_strategies, p=self._pi))

    def update(self, credits: np.ndarray) -> None:
        """Record per-strategy credits for this generation and update π.

        Parameters
        ----------
        credits : ndarray, shape (n_strategies,)
            Sum of credit values earned by each strategy this generation.
        """
        self._credit_history.append(credits.copy())

        # Sum over sliding window
        totals = np.zeros(self.n_strategies)
        for c in self._credit_history:
            totals += c

        if totals.sum() > 0:
            self._pi = totals / totals.sum()
            self._pi = np.maximum(self._pi, self.pi_min)
            self._pi /= self._pi.sum()  # re-normalise after floor
        else:
            self._pi[:] = 1.0 / self.n_strategies

    def reset(self) -> None:
        self._pi[:] = 1.0 / self.n_strategies
        self._credit_history.clear()
