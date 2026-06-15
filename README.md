# MOSADE

MOSADE (Multi-Objective Self-Adaptive Differential Evolution) is a Python
framework for multi-objective optimisation with real-valued decision variables.
It combines multiple differential-evolution mutation strategies,
decomposition-based environmental selection, strategy-level parameter memories,
and epsilon-constraint handling for constrained problems.

## Repository Contents

- `src/mosade/algorithm/`: MOSADE, differential-evolution strategies,
  decomposition, selection, adaptation, archive management, and baseline
  implementations.
- `src/mosade/problems/`: ZDT, DTLZ, WFG, DASCMOP, and real-world CRE benchmark
  problems.
- `src/mosade/metrics/`: hypervolume, IGD, IGD+, GD, spread, and spacing.
- `src/mosade/analysis/`: result merging, plotting, sensitivity analysis, and
  statistical utilities.
- `configs/`: reproducible experiment configurations.
- `scripts/`: command-line runners and result post-processing utilities.
- `tests/`: unit and smoke tests.
- `figures/`, `tables/`, `validation/`: generated outputs and summary data.

Raw per-run experiment directories are written to `results/` and are not tracked
by Git because complete benchmark runs can be large.

## Installation

```bash
pip install mosade
```

For the comparison baselines (pymoo) and the analysis/plotting tooling:

```bash
pip install "mosade[baselines,analysis]"
```

Python 3.10 or newer is required. To work on MOSADE itself, install from source
with the development tools:

```bash
git clone https://github.com/Levvvi/MOSADE.git
cd MOSADE
pip install -e ".[dev,analysis,baselines]"
```

## Quick Start

Run the default smoke experiment:

```bash
python scripts/run_experiment.py
```

Run a specific configuration:

```bash
python scripts/run_experiment.py --config configs/smoke_test.yaml
```

Run the test suite:

```bash
pytest
```

See `REPRODUCIBILITY.md` for the recommended workflow for reproducing generated
results.
