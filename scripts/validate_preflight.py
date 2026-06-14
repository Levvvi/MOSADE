"""Validate preflight results completeness."""
import json
import sys
from pathlib import Path

def validate(results_dir):
    results_dir = Path(results_dir)
    errors = []
    warnings = []

    # Find the preflight directory
    candidates = sorted(results_dir.glob("*preflight*"))
    if not candidates:
        print("ERROR: No preflight results found")
        sys.exit(1)
    run_dir = candidates[-1]
    print(f"Validating: {run_dir}")

    # Expected structure: run_dir / problem_dir / algo_dir / run_000 / files
    # OR: run_dir / problem_dir / run_000 / files (single algo mode)

    required_metrics = ["hv", "igd", "time_s", "n_evals", "n_solutions"]
    desired_metrics = ["igd_plus", "gd", "spread", "feasibility_ratio"]

    problem_dirs = [d for d in run_dir.iterdir()
                    if d.is_dir() and d.name not in ("plots", "tables")]

    if not problem_dirs:
        errors.append("No problem directories found")

    for pd in sorted(problem_dirs):
        # Check for algorithm subdirectories or direct run directories
        algo_dirs = [d for d in pd.iterdir()
                     if d.is_dir() and not d.name.startswith("run_")]
        run_dirs = [d for d in pd.iterdir()
                    if d.is_dir() and d.name.startswith("run_")]

        check_dirs = []
        if algo_dirs:
            for ad in algo_dirs:
                ad_runs = [d for d in ad.iterdir()
                          if d.is_dir() and d.name.startswith("run_")]
                for rd in ad_runs:
                    check_dirs.append((f"{pd.name}/{ad.name}", rd))
        elif run_dirs:
            for rd in run_dirs:
                check_dirs.append((pd.name, rd))
        else:
            errors.append(f"{pd.name}: no run directories found")
            continue

        for label, rd in check_dirs:
            # Check objectives.txt
            obj_path = rd / "objectives.txt"
            if not obj_path.exists():
                errors.append(f"{label}/{rd.name}: missing objectives.txt")

            # Check metrics.json
            met_path = rd / "metrics.json"
            if not met_path.exists():
                errors.append(f"{label}/{rd.name}: missing metrics.json")
            else:
                with open(met_path) as f:
                    m = json.load(f)
                for key in required_metrics:
                    if key not in m:
                        errors.append(f"{label}/{rd.name}: missing required metric '{key}'")
                    elif m[key] is None or (isinstance(m[key], float) and m[key] != m[key]):
                        errors.append(f"{label}/{rd.name}: metric '{key}' is null/NaN")
                for key in desired_metrics:
                    if key not in m:
                        warnings.append(f"{label}/{rd.name}: missing desired metric '{key}'")

                # Check HV is positive
                if "hv" in m and isinstance(m["hv"], (int, float)) and m["hv"] <= 0:
                    errors.append(f"{label}/{rd.name}: HV is {m['hv']} (should be > 0)")

            # Check history.json
            hist_path = rd / "history.json"
            if not hist_path.exists():
                warnings.append(f"{label}/{rd.name}: missing history.json")

    print(f"\n{'='*60}")
    print(f"ERRORS: {len(errors)}")
    for e in errors:
        print(f"  [ERROR] {e}")
    print(f"\nWARNINGS: {len(warnings)}")
    for w in warnings[:20]:  # cap at 20
        print(f"  [WARN] {w}")
    if len(warnings) > 20:
        print(f"  ... and {len(warnings)-20} more warnings")
    print(f"\n{'='*60}")
    if errors:
        print("PREFLIGHT FAILED - fix errors before proceeding")
        sys.exit(1)
    else:
        print("PREFLIGHT PASSED")

if __name__ == "__main__":
    validate(sys.argv[1] if len(sys.argv) > 1 else "results")
