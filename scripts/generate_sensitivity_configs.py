"""Generate parameter sensitivity config files for MOSADE.

For each parameter in [lp, memory_H, T_base_ratio, pi_min], produces one
YAML config per sweep value.  Every config runs MOSADE only (no baselines)
with one parameter varied and all others held at their defaults.

Output layout::

    configs/sensitivity/sens_{param}_{value}.yaml   (one per variant)
    configs/sensitivity/all_configs.txt             (manifest, one path per line)

Usage
-----
    # Full sensitivity run (n_runs=31, max_evals=100000)
    python scripts/generate_sensitivity_configs.py

    # Quick validation run (n_runs=3, max_evals=5000)
    python scripts/generate_sensitivity_configs.py --quick

    # Override output directory
    python scripts/generate_sensitivity_configs.py --output-dir configs/sensitivity

    # Generate only specific parameters
    python scripts/generate_sensitivity_configs.py --params lp memory_H
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Sweep specification
# ---------------------------------------------------------------------------

#: Defaults match MOSADE.__init__ signature exactly.
DEFAULTS: dict[str, int | float] = {
    "lp":           50,
    "memory_H":      5,
    "T_base_ratio":  0.10,
    "pi_min":        0.05,
}

#: Four sweep values per parameter.
SWEEP_VALUES: dict[str, list[int | float]] = {
    "lp":           [20, 50, 100, 200],
    "memory_H":     [3, 5, 10, 20],
    "T_base_ratio": [0.05, 0.10, 0.20, 0.30],
    "pi_min":       [0.01, 0.05, 0.10, 0.15],
}

#: Reduced problem set: representative unconstrained + constrained.
SENSITIVITY_PROBLEMS = [
    "ZDT1",
    "ZDT3",
    "WFG4",
    "WFG9",
    {"name": "DASCMOP7", "difficulty": 7},
]

#: Human-readable labels for tag/file names (floats need safe chars).
def _value_str(v: int | float) -> str:
    """Convert a parameter value to a filesystem-safe string."""
    if isinstance(v, float):
        return str(v).replace(".", "p")
    return str(v)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _build_config(
    param_name: str,
    param_value: int | float,
    n_runs: int,
    max_evals: int,
    track_interval: int,
    pop_size: int,
) -> dict:
    """Return a config dict for one (parameter, value) combination."""
    # Build the MOSADE algorithm block with the varied parameter.
    algo_block: dict = {
        "name": "MOSADE",
        "pop_size": pop_size,
        "max_evals": max_evals,
        "track_interval": track_interval,
    }
    # Inject all non-default parameters so the config is fully self-describing.
    for pname, default in DEFAULTS.items():
        value = param_value if pname == param_name else default
        algo_block[pname] = value

    tag = f"sens_{param_name}_{_value_str(param_value)}"
    return {
        "tag": tag,
        "results_dir": "results",
        "seed": 42,
        "n_runs": n_runs,
        "algorithms": [algo_block],
        "problems": SENSITIVITY_PROBLEMS,
    }


# ---------------------------------------------------------------------------
# YAML serialisation helper
# ---------------------------------------------------------------------------

class _LiteralStr(str):
    """Marker class so a string is emitted as a YAML literal block scalar."""


def _represent_literal(dumper: yaml.Dumper, data: _LiteralStr) -> yaml.Node:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


def _dump_config(cfg: dict) -> str:
    """Dump a config dict to a YAML string with readable formatting."""
    # Use a custom dumper that renders dicts inline when they're problem entries.
    class _Dumper(yaml.Dumper):
        pass

    def _represent_dict(dumper: yaml.Dumper, data: dict) -> yaml.Node:
        # Inline-style for small problem parameter dicts (e.g. {name: X, n_obj: 3})
        if all(isinstance(v, (str, int, float)) for v in data.values()) and len(data) <= 4:
            return dumper.represent_mapping(
                "tag:yaml.org,2002:map", data.items(), flow_style=True
            )
        return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())

    _Dumper.add_representer(dict, _represent_dict)
    return yaml.dump(cfg, Dumper=_Dumper, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------

def generate_configs(
    output_dir: Path,
    n_runs: int,
    max_evals: int,
    track_interval: int,
    pop_size: int,
    params: list[str] | None = None,
) -> list[Path]:
    """Generate all sensitivity config files and the manifest.

    Parameters
    ----------
    output_dir : Path
        Directory where ``sens_*.yaml`` files are written.
    n_runs : int
        Number of independent repetitions per config.
    max_evals : int
        Evaluation budget per run.
    track_interval : int
        HV/IGD snapshot interval for MOSADE.
    pop_size : int
        MOSADE population size.
    params : list of str or None
        Subset of parameters to generate; defaults to all four.

    Returns
    -------
    list of Path
        Paths to all generated config files (sorted by param then value).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    params = params or list(SWEEP_VALUES.keys())

    generated: list[Path] = []
    for param_name in params:
        if param_name not in SWEEP_VALUES:
            raise ValueError(
                f"Unknown parameter {param_name!r}. "
                f"Choose from: {list(SWEEP_VALUES)}"
            )
        for value in SWEEP_VALUES[param_name]:
            cfg = _build_config(param_name, value, n_runs, max_evals,
                                track_interval, pop_size)
            fname = f"sens_{param_name}_{_value_str(value)}.yaml"
            out_path = output_dir / fname
            out_path.write_text(_dump_config(cfg), encoding="utf-8")
            generated.append(out_path)
            print(f"  Written: {out_path}")

    # Write manifest
    manifest_path = output_dir / "all_configs.txt"
    manifest_path.write_text(
        "\n".join(str(p) for p in generated) + "\n",
        encoding="utf-8",
    )
    print(f"\nManifest written: {manifest_path}")
    print(f"Total configs: {len(generated)}")
    return generated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MOSADE parameter sensitivity configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/generate_sensitivity_configs.py
              python scripts/generate_sensitivity_configs.py --quick
              python scripts/generate_sensitivity_configs.py --params lp pi_min
        """),
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="configs/sensitivity",
        help="Output directory for generated YAML files (default: configs/sensitivity)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick-validation mode: n_runs=3, max_evals=5000 (overrides --n-runs / --max-evals)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=31,
        help="Independent repetitions per config (default: 31)",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=100_000,
        help="Function evaluation budget (default: 100000)",
    )
    parser.add_argument(
        "--track-interval",
        type=int,
        default=2000,
        help="HV/IGD snapshot interval for MOSADE (default: 2000)",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=100,
        help="MOSADE population size (default: 100)",
    )
    parser.add_argument(
        "--params",
        nargs="+",
        choices=list(SWEEP_VALUES.keys()),
        default=None,
        help="Parameters to sweep (default: all four)",
    )
    args = parser.parse_args()

    if args.quick:
        n_runs = 3
        max_evals = 5_000
        track_interval = 1_000
        pop_size = 50
        print("Quick-validation mode: n_runs=3, max_evals=5000, pop_size=50")
    else:
        n_runs = args.n_runs
        max_evals = args.max_evals
        track_interval = args.track_interval
        pop_size = args.pop_size

    generate_configs(
        output_dir=Path(args.output_dir),
        n_runs=n_runs,
        max_evals=max_evals,
        track_interval=track_interval,
        pop_size=pop_size,
        params=args.params,
    )


if __name__ == "__main__":
    main()
