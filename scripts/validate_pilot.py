"""Validate pilot results for performance sanity."""
import json
import sys
from pathlib import Path
import numpy as np

def validate_pilot(results_dir):
    results_dir = Path(results_dir)
    candidates = sorted(results_dir.glob("*pilot*"))
    if not candidates:
        print("ERROR: No pilot results found")
        sys.exit(1)
    run_dir = candidates[-1]
    print(f"Validating pilot: {run_dir}")

    issues = []
    info = []

    # Collect all metrics by algorithm and problem
    data = {}  # {problem: {algo: [hv_values]}}

    for pd in sorted(run_dir.iterdir()):
        if not pd.is_dir() or pd.name in ("plots", "tables"):
            continue
        prob_name = pd.name
        data[prob_name] = {}

        # Handle both nested (algo/run_000) and flat (run_000) layouts
        for sub in sorted(pd.iterdir()):
            if not sub.is_dir():
                continue
            if sub.name.startswith("run_"):
                # flat layout, single algo
                algo_name = "MOSADE"
                run_dirs = [sub]
            else:
                algo_name = sub.name
                run_dirs = sorted([d for d in sub.iterdir()
                                   if d.is_dir() and d.name.startswith("run_")])

            if algo_name not in data[prob_name]:
                data[prob_name][algo_name] = []

            for rd in run_dirs:
                mp = rd / "metrics.json"
                if mp.exists():
                    with open(mp) as f:
                        m = json.load(f)
                    if "hv" in m and m["hv"] is not None:
                        data[prob_name][algo_name].append(m["hv"])

    # Check MOSADE performance relative to others
    for prob, algos in sorted(data.items()):
        if "MOSADE" not in algos or not algos["MOSADE"]:
            issues.append(f"{prob}: MOSADE has no HV data")
            continue

        mosade_median = np.median(algos["MOSADE"])

        algo_medians = {}
        for algo, vals in algos.items():
            if vals:
                algo_medians[algo] = np.median(vals)

        if not algo_medians:
            continue

        best_algo = max(algo_medians, key=algo_medians.get)
        best_val = algo_medians[best_algo]

        # Check if MOSADE is catastrophically bad (< 50% of best)
        if best_val > 0 and mosade_median < 0.5 * best_val:
            issues.append(
                f"{prob}: MOSADE HV ({mosade_median:.4f}) is < 50% of best "
                f"({best_algo}: {best_val:.4f}) — possible bug"
            )

        rank = sorted(algo_medians.values(), reverse=True)
        mosade_rank = rank.index(mosade_median) + 1 if mosade_median in rank else "?"

        info.append(
            f"{prob}: MOSADE median HV = {mosade_median:.4f}, "
            f"rank {mosade_rank}/{len(algo_medians)}, "
            f"best = {best_algo} ({best_val:.4f})"
        )

    print(f"\n{'='*60}")
    print("PILOT PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    for line in info:
        print(f"  {line}")

    if issues:
        print(f"\nPERFORMANCE ISSUES ({len(issues)}):")
        for issue in issues:
            print(f"  [ISSUE] {issue}")
        print(f"\n{'='*60}")
        print("PILOT HAS ISSUES — investigate before full run")
        print("If MOSADE is catastrophically bad on a problem,")
        print("check the algorithm logic before wasting compute on 31 runs.")
    else:
        print(f"\n{'='*60}")
        print("PILOT PASSED — safe to proceed to full experiment")

if __name__ == "__main__":
    validate_pilot(sys.argv[1] if len(sys.argv) > 1 else "results")
