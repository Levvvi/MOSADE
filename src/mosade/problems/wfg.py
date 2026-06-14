"""WFG benchmark suite (Huband et al., 2006).

Scalable problems with diverse Pareto front geometries: convex, concave,
linear, disconnected, mixed, degenerate, multimodal, biased, non-separable.

All problems are bi-objective by default (M=2) with n_var=24 (standard
k=4 position parameters, l=20 distance parameters).

Reference: Huband, S., Hingston, P., Barone, L., While, L. (2006).
"A review of multiobjective test problems and a scalable test problem toolkit."
IEEE Trans. Evolutionary Computation, 10(5), 477-506.
"""

from __future__ import annotations

import numpy as np

from mosade.problems.base import Problem


# ======================================================================
# Shared WFG transformation and shape functions
# ======================================================================

def _correct_to_01(z: np.ndarray) -> np.ndarray:
    """Clamp to [0, 1] to handle floating point drift."""
    return np.clip(z, 0.0, 1.0)


# --- Transition functions ---

def _s_linear(y: np.ndarray, A: float) -> np.ndarray:
    return _correct_to_01(np.abs(y - A) / np.abs(np.floor(A - y) + A))


def _s_multi(y: np.ndarray, A: int, B: float, C: float) -> np.ndarray:
    tmp1 = np.abs(y - C) / (2.0 * (np.floor(C - y) + C))
    tmp2 = (4.0 * A + 2.0) * np.pi * (0.5 - tmp1)
    return _correct_to_01((1.0 + np.cos(tmp2) + 4.0 * B * tmp1**2) / (B + 2.0))


def _s_decept(y: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    tmp1 = np.floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B)
    tmp2 = np.floor(A + B - y) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B)
    return _correct_to_01(1.0 + (np.abs(y - A) - B) * (tmp1 + tmp2 + 1.0 / B))


def _b_flat(y: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    tmp1 = np.minimum(0.0, np.floor(y - B)) * A * (B - y) / B
    tmp2 = np.minimum(0.0, np.floor(C - y)) * (1.0 - A) * (y - C) / (1.0 - C)
    return _correct_to_01(A + tmp1 - tmp2)


def _b_poly(y: np.ndarray, alpha: float) -> np.ndarray:
    return _correct_to_01(y ** alpha)


def _b_param(
    y: np.ndarray,
    u: np.ndarray,
    A: float = 0.98 / 49.98,
    B: float = 0.02,
    C: float = 50.0,
) -> np.ndarray:
    """Parameter-dependent bias transformation (WFG reference, Section 2.3.5).

    FIX(audit B4): added to replace the dead-code loop in WFG7 that was
    immediately overwritten.  This implements the correct b_param formula from
    Huband et al. (2006).
    """
    v = A - (1.0 - 2.0 * u) * np.abs(np.floor(0.5 - u) + A)
    return _correct_to_01(y ** (B + (C - B) * v))


def _r_sum(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted sum reduction.  y: (N, k), w: (k,) -> (N,)"""
    return _correct_to_01(np.sum(y * w, axis=1) / np.sum(w))


def _r_nonsep(y: np.ndarray, A: int) -> np.ndarray:
    """Non-separable reduction.  y: (N, k) -> (N,)"""
    N, k = y.shape
    result = np.zeros(N)
    for j in range(k):
        result += y[:, j]
        for l in range(1, A):
            result += np.abs(y[:, j] - y[:, (j + l) % k])
    return _correct_to_01(result / (k / A * np.ceil(A / 2.0) * (1.0 + 2.0 * A - 2.0 * np.ceil(A / 2.0))))


# --- Shape functions ---

def _shape_linear(x: np.ndarray, M: int) -> np.ndarray:
    """Linear shape.  x: (N, M-1) -> (N, M)"""
    N = x.shape[0]
    h = np.ones((N, M))
    for m in range(M):
        if m < M - 1:
            for i in range(M - 1 - m):
                h[:, m] *= x[:, i]
        if m > 0:
            h[:, m] *= 1.0 - x[:, M - 1 - m]
    return h


def _shape_convex(x: np.ndarray, M: int) -> np.ndarray:
    N = x.shape[0]
    h = np.ones((N, M))
    for m in range(M):
        if m < M - 1:
            for i in range(M - 1 - m):
                h[:, m] *= 1.0 - np.cos(x[:, i] * np.pi / 2.0)
        if m > 0:
            h[:, m] *= 1.0 - np.sin(x[:, M - 1 - m] * np.pi / 2.0)
    return h


def _shape_concave(x: np.ndarray, M: int) -> np.ndarray:
    N = x.shape[0]
    h = np.ones((N, M))
    for m in range(M):
        if m < M - 1:
            for i in range(M - 1 - m):
                h[:, m] *= np.sin(x[:, i] * np.pi / 2.0)
        if m > 0:
            h[:, m] *= np.cos(x[:, M - 1 - m] * np.pi / 2.0)
    return h


def _shape_mixed(x1: np.ndarray, A: int, alpha: float) -> np.ndarray:
    """Mixed convex/concave shape for the first objective."""
    return _correct_to_01(
        (1.0 - x1 - np.cos(2.0 * A * np.pi * x1 + np.pi / 2.0) / (2.0 * A * np.pi)) ** alpha
    )


def _shape_disc(x1: np.ndarray, A: int, alpha: float, beta: float) -> np.ndarray:
    """Disconnected shape for the first objective."""
    return _correct_to_01(
        1.0 - x1 ** alpha * np.cos(A * x1 ** beta * np.pi) ** 2
    )


# ======================================================================
# Base WFG class
# ======================================================================

class _WFGBase(Problem):
    """Base class for WFG problems."""

    def __init__(self, n_var: int = 24, n_obj: int = 2, k: int | None = None) -> None:
        if k is None:
            # Use k = max(4, 2*(M-1)) as the reference default.
            # The old formula gave k=2 for M=3, but the Huband et al. reference
            # specifies k=2*(M-1), which is 4 for M=3 and always divisible by M-1.
            # max(4, ...) preserves the historical bi-objective default of k=4.
            k = max(4, 2 * (n_obj - 1))
        else:
            # User-supplied k: ensure divisibility by M-1.
            if n_obj > 1 and k % (n_obj - 1) != 0:
                k = max(n_obj - 1, k - (k % (n_obj - 1)))
        self._k = k
        self._l = n_var - k
        if self._l <= 0:
            raise ValueError(f"n_var ({n_var}) must be > k ({k})")
        lower = np.zeros(n_var)
        upper = np.arange(1, n_var + 1) * 2.0  # z_i_max = 2*i
        super().__init__(n_var=n_var, n_obj=n_obj, lower=lower, upper=upper)

    def _normalise_z(self, Z: np.ndarray) -> np.ndarray:
        """Normalise raw z to [0, 1] using z_max = 2*i."""
        z_max = np.arange(1, self.n_var + 1) * 2.0
        return Z / z_max

    def _compute_x(self, t: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Compute x from reduced t using degeneracy vector A."""
        N, M = t.shape[0], len(A) + 1
        x = np.zeros((N, M))
        for i in range(M - 1):
            x[:, i] = np.maximum(t[:, -1], A[i]) * (t[:, i] - 0.5) + 0.5
        x[:, -1] = t[:, -1]
        return x

    def _compute_objectives(self, x: np.ndarray, h: np.ndarray, S: np.ndarray) -> np.ndarray:
        """f_m = x_M * D + S_m * h_m."""
        M = h.shape[1]
        D = 1.0
        f = np.zeros_like(h)
        for m in range(M):
            f[:, m] = x[:, -1] * D + S[m] * h[:, m]
        return f


# ======================================================================
# WFG1 - WFG9
# ======================================================================

class WFG1(_WFGBase):
    """Separable, uni-modal.  Convex + mixed PF shape.  Biased."""

    def _evaluate(self, X: np.ndarray):
        y = self._normalise_z(X)
        N, k, l, M = y.shape[0], self._k, self._l, self.n_obj

        # t1: bias (polynomial for distance params only)
        t1 = y.copy()
        t1[:, k:] = _b_poly(y[:, k:], 0.02)

        # t2: bias (flat for position params only)
        t2 = t1.copy()
        for i in range(k):
            t2[:, i] = _b_flat(t1[:, i], 0.8, 0.75, 0.85)

        # t3: bias (polynomial)
        t3 = _b_poly(t2, 50.0)

        # t4: reduction (weighted sum)
        t4 = np.zeros((N, M))
        w = np.arange(1, self.n_var + 1, dtype=float) * 2.0
        for i in range(M - 1):
            start = i * k // (M - 1)
            end = (i + 1) * k // (M - 1)
            t4[:, i] = _r_sum(t3[:, start:end], w[start:end])
        t4[:, -1] = _r_sum(t3[:, k:], w[k:])

        # Shape
        A = np.ones(M - 1)
        x = self._compute_x(t4, A)
        h = np.zeros((N, M))
        h[:, 0] = _shape_convex(x[:, :M-1], M)[:, 0]
        for m in range(1, M - 1):
            h[:, m] = _shape_convex(x[:, :M-1], M)[:, m]
        h[:, -1] = _shape_mixed(x[:, 0], 5, 1.0)

        S = np.arange(1, M + 1) * 2.0
        return self._compute_objectives(x, h, S), None


class WFG2(_WFGBase):
    """Non-separable, uni-modal.  Convex + disconnected PF."""

    def _evaluate(self, X: np.ndarray):
        y = self._normalise_z(X)
        N, k, l, M = y.shape[0], self._k, self._l, self.n_obj

        # t1: shift (linear for distance params)
        t1 = y.copy()
        t1[:, k:] = _s_linear(y[:, k:], 0.35)

        # t2: non-separable reduction on distance params (pairs)
        t2_pos = t1[:, :k].copy()
        l_half = l // 2
        t2_dist = np.zeros((N, l_half))
        for i in range(l_half):
            t2_dist[:, i] = _r_nonsep(t1[:, k + 2*i:k + 2*i + 2], 2)
        t2 = np.column_stack([t2_pos, t2_dist])

        # t3: weighted sum reduction
        n_t2 = t2.shape[1]
        t3 = np.zeros((N, M))
        w = np.ones(n_t2)
        items_per_group = k // (M - 1)
        for i in range(M - 1):
            start = i * items_per_group
            end = (i + 1) * items_per_group
            t3[:, i] = _r_sum(t2[:, start:end], w[start:end])
        t3[:, -1] = _r_sum(t2[:, k:], w[k:n_t2] if k < n_t2 else np.ones(t2[:, k:].shape[1]))

        A = np.ones(M - 1)
        x = self._compute_x(t3, A)
        h = np.zeros((N, M))
        h_convex = _shape_convex(x[:, :M-1], M)
        for m in range(M - 1):
            h[:, m] = h_convex[:, m]
        h[:, -1] = _shape_disc(x[:, 0], 5, 1.0, 1.0)

        S = np.arange(1, M + 1) * 2.0
        return self._compute_objectives(x, h, S), None


class WFG3(_WFGBase):
    """Non-separable, uni-modal.  Linear, degenerate PF."""

    def _evaluate(self, X: np.ndarray):
        y = self._normalise_z(X)
        N, k, l, M = y.shape[0], self._k, self._l, self.n_obj

        t1 = y.copy()
        t1[:, k:] = _s_linear(y[:, k:], 0.35)

        l_half = l // 2
        t2_pos = t1[:, :k].copy()
        t2_dist = np.zeros((N, l_half))
        for i in range(l_half):
            t2_dist[:, i] = _r_nonsep(t1[:, k + 2*i:k + 2*i + 2], 2)
        t2 = np.column_stack([t2_pos, t2_dist])

        n_t2 = t2.shape[1]
        t3 = np.zeros((N, M))
        w = np.ones(n_t2)
        items_per_group = k // (M - 1)
        for i in range(M - 1):
            start = i * items_per_group
            end = (i + 1) * items_per_group
            t3[:, i] = _r_sum(t2[:, start:end], w[start:end])
        t3[:, -1] = _r_sum(t2[:, k:], w[k:n_t2] if k < n_t2 else np.ones(t2[:, k:].shape[1]))

        # Degenerate: A = [1, 0, 0, ..., 0]
        A = np.zeros(M - 1)
        A[0] = 1.0
        x = self._compute_x(t3, A)
        h = _shape_linear(x[:, :M-1], M)

        S = np.arange(1, M + 1) * 2.0
        return self._compute_objectives(x, h, S), None


class WFG4(_WFGBase):
    """Separable, multi-modal.  Concave PF."""

    def _evaluate(self, X: np.ndarray):
        y = self._normalise_z(X)
        N, k, l, M = y.shape[0], self._k, self._l, self.n_obj

        # t1: multi-modal shift
        t1 = _s_multi(y, 30, 10.0, 0.35)

        # t2: weighted sum reduction
        t2 = np.zeros((N, M))
        w = np.ones(self.n_var)
        items_per_group = k // (M - 1)
        for i in range(M - 1):
            start = i * items_per_group
            end = (i + 1) * items_per_group
            t2[:, i] = _r_sum(t1[:, start:end], w[start:end])
        t2[:, -1] = _r_sum(t1[:, k:], w[k:])

        A = np.ones(M - 1)
        x = self._compute_x(t2, A)
        h = _shape_concave(x[:, :M-1], M)

        S = np.arange(1, M + 1) * 2.0
        return self._compute_objectives(x, h, S), None


class WFG5(_WFGBase):
    """Separable, deceptive.  Concave PF."""

    def _evaluate(self, X: np.ndarray):
        y = self._normalise_z(X)
        N, k, l, M = y.shape[0], self._k, self._l, self.n_obj

        t1 = _s_decept(y, 0.35, 0.001, 0.05)

        t2 = np.zeros((N, M))
        w = np.ones(self.n_var)
        items_per_group = k // (M - 1)
        for i in range(M - 1):
            start = i * items_per_group
            end = (i + 1) * items_per_group
            t2[:, i] = _r_sum(t1[:, start:end], w[start:end])
        t2[:, -1] = _r_sum(t1[:, k:], w[k:])

        A = np.ones(M - 1)
        x = self._compute_x(t2, A)
        h = _shape_concave(x[:, :M-1], M)

        S = np.arange(1, M + 1) * 2.0
        return self._compute_objectives(x, h, S), None


class WFG6(_WFGBase):
    """Non-separable, uni-modal.  Concave PF."""

    def _evaluate(self, X: np.ndarray):
        y = self._normalise_z(X)
        N, k, l, M = y.shape[0], self._k, self._l, self.n_obj

        t1 = y.copy()
        t1[:, k:] = _s_linear(y[:, k:], 0.35)

        t2 = np.zeros((N, M))
        items_per_group = k // (M - 1)
        for i in range(M - 1):
            start = i * items_per_group
            end = (i + 1) * items_per_group
            t2[:, i] = _r_nonsep(t1[:, start:end], items_per_group)
        t2[:, -1] = _r_nonsep(t1[:, k:], l)

        A = np.ones(M - 1)
        x = self._compute_x(t2, A)
        h = _shape_concave(x[:, :M-1], M)

        S = np.arange(1, M + 1) * 2.0
        return self._compute_objectives(x, h, S), None


class WFG7(_WFGBase):
    """Separable, uni-modal.  Concave PF.  Parameter-dependent bias."""

    def _evaluate(self, X: np.ndarray):
        y = self._normalise_z(X)
        N, k, l, M = y.shape[0], self._k, self._l, self.n_obj

        # FIX(audit B4): implement WFG7's parameter-dependent bias correctly.
        # The old code had a loop whose output was immediately overwritten by a
        # batch assignment (the loop was dead code with an `if False else` guard).
        # WFG7 t1: for each position parameter i, bias depends on a weighted sum
        # of the *remaining* parameters y_{i+1:} (Huband et al. 2006, Alg 2.3.6).
        t1 = y.copy()
        w = np.ones(self.n_var)
        for i in range(k):
            if i + 1 < self.n_var:
                u = _r_sum(y[:, i + 1:], w[i + 1:])
            else:
                u = np.full(N, 0.5)  # no remaining params: use neutral value
            t1[:, i] = _b_param(y[:, i], u)
        t1[:, k:] = _s_linear(y[:, k:], 0.35)

        t2 = np.zeros((N, M))
        items_per_group = k // (M - 1)
        for i in range(M - 1):
            start = i * items_per_group
            end = (i + 1) * items_per_group
            t2[:, i] = _r_sum(t1[:, start:end], w[start:end])
        t2[:, -1] = _r_sum(t1[:, k:], w[k:])

        A = np.ones(M - 1)
        x = self._compute_x(t2, A)
        h = _shape_concave(x[:, :M-1], M)

        S = np.arange(1, M + 1) * 2.0
        return self._compute_objectives(x, h, S), None


class WFG8(_WFGBase):
    """Non-separable, uni-modal.  Concave PF.  Parameter-dependent bias."""

    def _evaluate(self, X: np.ndarray):
        y = self._normalise_z(X)
        N, k, l, M = y.shape[0], self._k, self._l, self.n_obj

        # t1: parameter-dependent bias for distance params (b_param, not b_poly)
        t1 = y.copy()
        w = np.ones(self.n_var)
        for i in range(k, self.n_var):
            u = _r_sum(y[:, :i], w[:i])
            t1[:, i] = _b_param(y[:, i], u)

        t1[:, k:] = _s_linear(t1[:, k:], 0.35)

        t2 = np.zeros((N, M))
        items_per_group = k // (M - 1)
        for i in range(M - 1):
            start = i * items_per_group
            end = (i + 1) * items_per_group
            t2[:, i] = _r_nonsep(t1[:, start:end], items_per_group)
        t2[:, -1] = _r_nonsep(t1[:, k:], l)

        A = np.ones(M - 1)
        x = self._compute_x(t2, A)
        h = _shape_concave(x[:, :M-1], M)

        S = np.arange(1, M + 1) * 2.0
        return self._compute_objectives(x, h, S), None


class WFG9(_WFGBase):
    """Non-separable, multi-modal, deceptive.  Concave PF."""

    def _evaluate(self, X: np.ndarray):
        y = self._normalise_z(X)
        N, k, l, M = y.shape[0], self._k, self._l, self.n_obj

        # t1: parameter-dependent bias (b_param; shift depends on later params)
        t1 = y.copy()
        w = np.ones(self.n_var)
        for i in range(self.n_var - 1):
            u = _r_sum(y[:, i+1:], w[i+1:])
            t1[:, i] = _b_param(y[:, i], u)

        # t2: shift (multi-modal for position, deceptive for distance)
        t2 = t1.copy()
        t2[:, :k] = _s_multi(t1[:, :k], 30, 95.0, 0.35)
        t2[:, k:] = _s_decept(t1[:, k:], 0.35, 0.001, 0.05)

        # t3: non-separable reduction
        t3 = np.zeros((N, M))
        items_per_group = k // (M - 1)
        for i in range(M - 1):
            start = i * items_per_group
            end = (i + 1) * items_per_group
            t3[:, i] = _r_nonsep(t2[:, start:end], items_per_group)
        t3[:, -1] = _r_nonsep(t2[:, k:], l)

        A = np.ones(M - 1)
        x = self._compute_x(t3, A)
        h = _shape_concave(x[:, :M-1], M)

        S = np.arange(1, M + 1) * 2.0
        return self._compute_objectives(x, h, S), None
