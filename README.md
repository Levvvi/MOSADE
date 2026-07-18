# MOSADE

[![CI](https://github.com/Levvvi/MOSADE/actions/workflows/ci.yml/badge.svg)](https://github.com/Levvvi/MOSADE/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mosade.svg)](https://pypi.org/project/mosade/)
[![Python](https://img.shields.io/pypi/pyversions/mosade.svg)](https://pypi.org/project/mosade/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Multi-Objective Self-Adaptive Differential Evolution** — a Python library for
multi-objective optimisation over real-valued decision variables, including
constrained problems.

MOSADE combines several differential-evolution mutation strategies under an
online, self-adaptive credit-assignment scheme, decomposition-based
environmental selection, per-strategy parameter memories, and ε-constraint
handling for constrained problems. It ships with the standard ZDT, DTLZ and WFG
suites, the constrained DAS-CMOP suite, and real-world CRE problems, together
with the usual quality indicators (HV, IGD, IGD+, GD, spread).

It also **interoperates with [pymoo](https://pymoo.org)**: the comparison
baselines are pymoo's own algorithm implementations, run on MOSADE's problems
and scored by the same indicators, using the same problem and metric interfaces
to reduce implementation differences rather than relying on re-implemented
competitors.

## Installation

Install the latest released package from PyPI:

```bash
python -m pip install mosade
```

Python 3.10 or newer is required. Install the optional analysis and pymoo
baseline dependencies with:

```bash
python -m pip install "mosade[analysis,baselines]"
```

For development from source:

```bash
git clone https://github.com/Levvvi/MOSADE.git
cd MOSADE
python -m pip install -e ".[dev,analysis,baselines]"
```

## Quick start

```python
import numpy as np
from mosade.problems import ZDT1
from mosade.algorithm import MOSADE
from mosade.metrics import hypervolume, igd

# Pick a benchmark problem (or subclass mosade.problems.Problem with your own).
problem = ZDT1(n_var=30)

# Run MOSADE. A fixed seed makes the run deterministic within the pinned, tested environment. (~20 s)
result = MOSADE(pop_size=100, max_evals=25_000, seed=0).run(problem)

print("non-dominated solutions:", result.F.shape)   # (200, 2)
print("function evaluations:   ", result.n_evals)    # 25030

# Score the approximation set against the analytical Pareto front.
pf = problem.pareto_front(200)
print("hypervolume (ref 1.1):  ", round(hypervolume(result.F, ref=np.array([1.1, 1.1])), 4))  # ≈ 0.854
print("IGD:                    ", round(igd(result.F, pf), 4))                                  # ≈ 0.013
```

`result.F` and `result.X` are the objective and decision vectors of the final
non-dominated set; `result.history` holds per-generation diagnostics (strategy
usage, parameter memories, convergence snapshots).

To run a configured experiment from the command line:

```bash
python scripts/run_experiment.py --config configs/smoke_test.yaml
```

## What's inside

- **Algorithms** (`mosade.algorithm`): `MOSADE`, plus `NSGA2` and `MOEAD`
  reference implementations and a `PymooAlgorithm` adapter that runs pymoo's
  algorithms as comparison baselines on MOSADE's own problems.
- **Problems** (`mosade.problems`): ZDT1–4 and ZDT6, DTLZ1–4 and DTLZ7,
  WFG1–9, the constrained DAS-CMOP1–9 suite, and real-world CRE problems — all
  sharing one `Problem` interface with a `g(x) <= 0` feasibility convention.
- **Metrics** (`mosade.metrics`): `hypervolume`, `igd`, `igd_plus`, `gd`,
  `spread`.
- **Reproducibility**: per-run seeding via NumPy's `SeedSequence` (statistically
  independent streams, not offset base seeds). See `REPRODUCIBILITY.md`.

## Benchmarking and reproduction

The full experiment harness — multi-run benchmarking, statistical comparison
(Wilcoxon signed-rank with Holm correction and Vargha–Delaney A12 effect
sizes), and figure/table generation — lives under `experiments/` (the
`mosade_experiments` package) and is intentionally kept out of the installed
library. See `REPRODUCIBILITY.md` for the recommended workflow.

## Testing and coverage

```bash
pytest                                               # full suite
pytest --cov=src/mosade --cov-report=term-missing    # with coverage
```

On the 0.1.1 release commit, a clean test run reports 287 passed and 5 skipped.
Line coverage for the shipped library (`src/mosade`) is 89%, with CI enforcing
at least 85% on Python 3.10, 3.11, 3.12, and 3.13.

The suite emphasises what matters for a stochastic optimiser: seed-determinism,
structural invariants (population size, decision bounds, constraint
feasibility), indicator correctness against analytically known Pareto fronts,
and regression snapshots — not line coverage for its own sake.

## Project layout

```
src/mosade/     # the installable library: algorithm, problems, metrics, runner
experiments/    # thesis benchmarking and analysis tooling (not packaged)
tests/          # library test suite
configs/        # example experiment configurations
```

## License

Released under the MIT License — see [LICENSE](LICENSE).
