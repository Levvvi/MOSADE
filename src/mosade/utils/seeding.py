"""Deterministic seed handling for reproducible experiments."""

from __future__ import annotations

import numpy as np


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Create a NumPy Generator from a seed.

    Parameters
    ----------
    seed : int or None
        If None, a non-deterministic seed is used.

    Returns
    -------
    np.random.Generator
        A PCG64-backed Generator instance.
    """
    return np.random.default_rng(seed)


def seed_sequence(base_seed: int, count: int) -> list[int]:
    """Generate *count* independent child seeds from *base_seed*.

    Useful for producing per-run seeds from a single experiment seed.
    """
    ss = np.random.SeedSequence(base_seed)
    children = ss.spawn(count)
    return [int(c.generate_state(1)[0]) for c in children]
