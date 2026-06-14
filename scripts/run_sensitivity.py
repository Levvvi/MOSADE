"""Run all sensitivity analysis configs and collect results.

Reads a manifest file (``configs/sensitivity/all_configs.txt``) produced by
``generate_sensitivity_configs.py``, runs each config via the standard
experiment runner, and writes a JSON manifest mapping each
``(param_name, param_value)`` pair to its result directory.

Usage
-----
    # Run all configs listed in the default manifest
    python scripts/run_sensitivity.py

    # Run configs listed in a specific manifest
    python scripts/run_sensitivity.py --manifest configs/sensitivity/all_configs.txt

    # Dry-run: print the commands that would be executed without running them
    python scripts/run_sensitivity.py --dry-run

    # Write the result manifest to a custom path
    python scripts/run_sensitivity.py --output results/sensitivity_manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the src/ package is importable when executed as a script.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mosade.runner.experiment import run_experiment  # noqa: E402
from mosade.utils.io import load_config, save_json  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_tag(config_path: Path) -> tuple[str, str] | None:
    """Extract (param_name, value_str) from a sensitivity config filename.

    Filenames follow the pattern ``sens_{param}_{value}.yaml``.
    Returns None if the name does not match.
    """
    m = re.match(r"sens_(.+?)_([\w.]+)\.yaml$", config_path.name)
    if m:
        return m.group(1), m.group(2)
    return None


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_sensitivity(
    manifest_path: Path,
    output_path: Path,
    dry_run: bool = False,
) -> dict:
    """Run all configs in *manifest_path* and return the result manifest dict.

    Parameters
    ----------
    manifest_path : Path
        Path to a text file with one config path per line.
    output_path : Path
        Where to write the JSON result manifest.
    dry_run : bool
        If True, print commands but do not run them.

    Returns
    -------
    dict
        Nested dict: ``{param_name: {value_str: result_dir_str}}``.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run generate_sensitivity_configs.py first."
        )

    config_paths = [
        Path(line.strip())
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    print(f"Found {len(config_paths)} configs in manifest.")

    result_manifest: dict[str, dict[str, str]] = {}
    total_start = time.perf_counter()

    for i, cfg_path in enumerate(config_paths, 1):
        parsed = _parse_tag(cfg_path)
        param_name, value_str = parsed if parsed else ("unknown", cfg_path.stem)

        print(f"\n[{i}/{len(config_paths)}] {cfg_path.name}  (param={param_name}, value={value_str})")

        if dry_run:
            print(f"  [DRY RUN] python scripts/run_experiment.py --config {cfg_path}")
            continue

        if not cfg_path.exists():
            print(f"  WARNING: config file not found — skipping: {cfg_path}")
            continue

        t0 = time.perf_counter()
        result_dir = run_experiment(cfg_path)
        elapsed = time.perf_counter() - t0
        print(f"  Done in {elapsed:.1f}s.  Results: {result_dir}")

        if param_name not in result_manifest:
            result_manifest[param_name] = {}
        result_manifest[param_name][value_str] = str(result_dir)

    total_elapsed = time.perf_counter() - total_start

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_manifest, f, indent=2)
        print(f"\nResult manifest written: {output_path}")
        print(f"Total wall time: {total_elapsed:.1f}s")

    return result_manifest


def run_sensitivity_config(
    config_path: Path,
    run_dir: Path,
    workers: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Run the A1-A7 formal sensitivity config schema.

    The config contains one ``base_algorithm`` and a ``scan`` section with
    low/default/high values.  Each setting is run as a separate standard
    experiment under ``run_dir``.
    """
    cfg = load_config(config_path)
    base = dict(cfg["base_algorithm"])
    scan = list(cfg["scan"])
    problems = cfg["problems"]
    seed = cfg.get("seed", 42)
    n_runs = cfg.get("n_runs", 11)
    results_dir = str(run_dir)
    manifest: dict[str, dict[str, str]] = {}
    run_dir.mkdir(parents=True, exist_ok=True)

    for item in scan:
        param = item["parameter"]
        manifest[param] = {}
        for label, value in item["values"].items():
            algo = dict(base)
            algo["name"] = f"MOSADE_{param}_{label}"
            algo[param] = value
            child_cfg = {
                "tag": f"e3_{param}_{label}",
                "results_dir": results_dir,
                "seed": seed,
                "parallel_workers": workers if workers is not None else cfg.get("parallel_workers", 1),
                "n_runs": n_runs,
                "algorithms": [algo],
                "problems": problems,
            }
            child_config_path = run_dir / "_configs" / f"e3_{param}_{label}.yaml"
            child_config_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(child_config_path.with_suffix(".json"), child_cfg)
            yaml_text = _to_simple_yaml(child_cfg)
            child_config_path.write_text(yaml_text, encoding="utf-8")
            child_run_dir = run_dir / f"{param}_{label}"
            print(f"[E3] {param}={value} ({label}) -> {child_run_dir}")
            if dry_run:
                print(f"  [DRY RUN] {child_config_path}")
            else:
                run_experiment(child_config_path, run_dir=child_run_dir, workers=workers)
            manifest[param][label] = str(child_run_dir)

    manifest_path = run_dir / "sensitivity_result_manifest.json"
    save_json(manifest_path, manifest)
    print(f"Result manifest written: {manifest_path}")
    return manifest


def _to_simple_yaml(obj: object, indent: int = 0) -> str:
    """Serialise the limited config structures used by this script."""
    sp = " " * indent
    if isinstance(obj, dict):
        lines = []
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{sp}{key}:")
                lines.append(_to_simple_yaml(value, indent + 2))
            elif isinstance(value, str):
                lines.append(f'{sp}{key}: "{value}"')
            else:
                lines.append(f"{sp}{key}: {value}")
        return "\n".join(lines)
    if isinstance(obj, list):
        lines = []
        for item in obj:
            if isinstance(item, dict):
                lines.append(f"{sp}-")
                lines.append(_to_simple_yaml(item, indent + 2))
            elif isinstance(item, str):
                lines.append(f'{sp}- "{item}"')
            else:
                lines.append(f"{sp}- {item}")
        return "\n".join(lines)
    return f"{sp}{obj}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all sensitivity analysis configs and collect results."
    )
    parser.add_argument(
        "--manifest", "-m",
        default="configs/sensitivity/all_configs.txt",
        help="Path to the manifest file (default: configs/sensitivity/all_configs.txt)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="A1-A7 formal sensitivity YAML config with base_algorithm and scan sections",
    )
    parser.add_argument(
        "--run-dir",
        default="results/e3_sensitivity_design_constants",
        help="Parent result directory for --config mode",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker override for --config mode",
    )
    parser.add_argument(
        "--output", "-o",
        default="results/sensitivity_manifest.json",
        help="Where to write the result manifest JSON (default: results/sensitivity_manifest.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them",
    )
    args = parser.parse_args()

    if args.config:
        run_sensitivity_config(
            config_path=Path(args.config),
            run_dir=Path(args.run_dir),
            workers=args.workers,
            dry_run=args.dry_run,
        )
    else:
        run_sensitivity(
            manifest_path=Path(args.manifest),
            output_path=Path(args.output),
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
