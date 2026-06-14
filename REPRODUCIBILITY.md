# Reproducibility

This repository is structured so that another user can install the same Python
package, run the provided configurations, and regenerate experiment outputs.

## Environment

Use Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,analysis]"
```

On Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

## Validation

Run the test suite before launching longer experiments:

```bash
pytest
```

If the default system temporary directory has permission issues on Windows, use
a repository-local temporary directory:

```bash
pytest --basetemp tmp\pytest
```

## Running Experiments

Small validation run:

```bash
python scripts/run_experiment.py --config configs/smoke_test.yaml
```

Benchmark examples:

```bash
python scripts/run_experiment.py --config configs/benchmark_zdt.yaml
python scripts/run_experiment.py --config configs/benchmark_wfg.yaml
python scripts/run_experiment.py --config configs/benchmark_dascmop.yaml
```

Experiment outputs are created under `results/`. Each configuration contains
the seed, algorithm parameters, benchmark list, and output directory settings
needed to regenerate the corresponding run structure.

## Generated Outputs

The repository includes compact generated outputs in `figures/`, `tables/`, and
`validation/`. Complete raw per-run results are intentionally excluded from Git
because they can be large. To regenerate them, run the matching configuration in
`configs/`.
