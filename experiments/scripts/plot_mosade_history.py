"""Quick visual inspection for a single MOSADE history.json file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _plot_lines(x, ys, labels, title, ylabel, outpath: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    for y, label in zip(ys, labels):
        plt.plot(x, y, label=label)
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("history_json", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("history_plots"))
    args = ap.parse_args()

    hist = _load(args.history_json)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    gen = hist.get("gen", list(range(1, len(hist.get("strategy_probs", [])) + 1)))

    if "strategy_probs" in hist and hist["strategy_probs"]:
        arr = np.asarray(hist["strategy_probs"], dtype=float)
        _plot_lines(gen, [arr[:, i] for i in range(arr.shape[1])],
                    [f"S{i+1}" for i in range(arr.shape[1])],
                    "Strategy probabilities", "Probability",
                    outdir / "strategy_probs.png")

    if "strategy_use_counts" in hist and hist["strategy_use_counts"]:
        arr = np.asarray(hist["strategy_use_counts"], dtype=float)
        _plot_lines(gen, [arr[:, i] for i in range(arr.shape[1])],
                    [f"S{i+1}" for i in range(arr.shape[1])],
                    "Strategy usage counts", "Count",
                    outdir / "strategy_usage.png")

    if "strategy_credit_total" in hist and hist["strategy_credit_total"]:
        arr = np.asarray(hist["strategy_credit_total"], dtype=float)
        _plot_lines(gen, [arr[:, i] for i in range(arr.shape[1])],
                    [f"S{i+1}" for i in range(arr.shape[1])],
                    "Strategy credit", "Credit",
                    outdir / "strategy_credit.png")

    if "delta" in hist and "T" in hist:
        plt.figure(figsize=(8, 4.5))
        plt.plot(gen, hist["delta"], label="delta")
        plt.plot(gen, hist["T"], label="T")
        if "archive_size" in hist:
            plt.plot(gen, hist["archive_size"], label="archive_size")
        plt.xlabel("Generation")
        plt.ylabel("Value")
        plt.title("Search-scope telemetry")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "search_scope.png", dpi=200)
        plt.close()

    if "epsilon" in hist and hist["epsilon"]:
        plt.figure(figsize=(8, 4.5))
        plt.plot(gen, hist["epsilon"], label="epsilon")
        if "feasibility_ratio" in hist:
            plt.plot(gen, hist["feasibility_ratio"], label="feasibility_ratio")
        if "best_cv" in hist:
            plt.plot(gen, hist["best_cv"], label="best_cv")
        plt.xlabel("Generation")
        plt.ylabel("Value")
        plt.title("Constraint-handling telemetry")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "epsilon_feasibility.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
