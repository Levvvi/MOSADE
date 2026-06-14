"""Validation script for DASCMOP feasibility ratios.

Samples random solutions and reports the fraction that are feasible at each
difficulty level.  Verifies that the constraint formulations produce the
expected difficulty ordering.

Usage:
    python scripts/validate_dascmop.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from mosade.problems.dascmop import (
    DASCMOP1, DASCMOP2, DASCMOP3, DASCMOP4, DASCMOP5, DASCMOP6,
    DASCMOP7, DASCMOP8, DASCMOP9, DIFFICULTY_SETTINGS,
)

CLASSES = [DASCMOP1, DASCMOP2, DASCMOP3, DASCMOP4, DASCMOP5, DASCMOP6,
           DASCMOP7, DASCMOP8, DASCMOP9]

N_SAMPLES = 10_000
SEED = 42


def feasibility_ratio(cls, level: int, rng: np.random.Generator) -> float:
    """Return fraction of random solutions that are feasible."""
    p = cls(n_var=30, difficulty=level)
    X = rng.random((N_SAMPLES, 30))
    _, CV = p.evaluate(X)
    return float(np.sum(CV == 0.0) / N_SAMPLES)


def main() -> None:
    rng = np.random.default_rng(SEED)

    # --- User-requested levels {1, 7, 13} ---
    print("=" * 72)
    print("Feasibility ratios at difficulty levels {1, 7, 13}")
    print("  Level 1:  (eta=0.25, zeta=0,   gamma=0)   -- eta only")
    print("  Level 7:  (eta=0,    zeta=0,   gamma=0.5) -- gamma only")
    print("  Level 13: (eta=0,    zeta=1.0, gamma=0)   -- extreme zeta")
    print("=" * 72)
    print(f"{'Problem':<12} | {'Level 1':>10} | {'Level 7':>10} | {'Level 13':>10}")
    print("-" * 55)

    for cls in CLASSES:
        ratios = [feasibility_ratio(cls, lv, rng) for lv in [1, 7, 13]]
        print(f"{cls.__name__:<12} | {ratios[0]:>9.4f} | {ratios[1]:>9.4f} | {ratios[2]:>9.4f}")

    print()
    print("Note: Level 7 > Level 1 is expected -- these levels test different")
    print("constraint axes (eta, gamma, zeta), not progressively harder overall.")
    print()

    # --- Eta-only levels {1, 5, 9} (should be strictly monotone) ---
    print("=" * 72)
    print("Feasibility ratios at eta-only levels {1, 5, 9}")
    print("  Level 1: eta=0.25,  Level 5: eta=0.5,  Level 9: eta=0.75")
    print("  (zeta=0, gamma=0 for all three -- only eta constraint active)")
    print("=" * 72)
    print(f"{'Problem':<12} | {'Level 1':>10} | {'Level 5':>10} | {'Level 9':>10} | {'Monotone?':>10}")
    print("-" * 65)

    all_monotone = True
    for cls in CLASSES:
        ratios = [feasibility_ratio(cls, lv, rng) for lv in [1, 5, 9]]
        mono = ratios[0] >= ratios[1] >= ratios[2]
        all_monotone = all_monotone and mono
        tag = "YES" if mono else "NO !!!"
        print(f"{cls.__name__:<12} | {ratios[0]:>9.4f} | {ratios[1]:>9.4f} | {ratios[2]:>9.4f} | {tag:<10}")

    print()
    if all_monotone:
        print("PASS: All problems show monotonically decreasing feasibility")
        print("      as eta increases (levels 1 -> 5 -> 9).")
    else:
        print("FAIL: Some problems violate monotonicity!")
        sys.exit(1)


if __name__ == "__main__":
    main()
