# -*- coding: utf-8 -*-
"""
Evaluation — Import AI Studio Auto Model results.

"""
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SHARED = os.path.join(PROJECT_ROOT, "shared")
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

from paths import *  # noqa: E402,F403
ensure_dirs()

import matplotlib.pyplot as plt
import pandas as pd

# Paths (defaults from shared/paths.py)
DEFAULT_CSV = AI_STUDIO_CSV      # data/Results_All.csv
OUT_JSON = AI_STUDIO_JSON        # outputs/json/ai_studio_results.json
FIG_DIR = FIGURES_DIR            # outputs/figures/


def parse_results(csv_path: str) -> dict:
    """Read Results_All.csv and build a JSON-friendly summary per model."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    models = []
    for _, row in df.iterrows():
        name = str(row["Model"]).strip()
        err = row.get("Classification Error")

        # Rows with no error value = Auto Model did not finish that model
        if pd.isna(err) or str(err).strip() == "":
            models.append({
                "model": name,
                "classification_error": None,
                "classification_error_std": None,
                "accuracy": None,
                "status": "not_completed",
            })
            continue

        err = float(err)
        std = row.get("Standard Deviation")
        std = float(std) if pd.notna(std) else None
        models.append({
            "model": name,
            "classification_error": err,
            "classification_error_std": std,
            "accuracy": round(1.0 - err, 4),
            "status": "completed",
        })

    completed = [m for m in models if m["status"] == "completed"]
    best = min(completed, key=lambda m: m["classification_error"]) if completed else None

    return {
        "source_file": os.path.basename(csv_path),
        "n_completed": len(completed),
        "n_total": len(models),
        "models": models,
        "best_model": best["model"] if best else None,
        "best_accuracy": best["accuracy"] if best else None,
        "best_classification_error": best["classification_error"] if best else None,
        "notes": (
            "Auto Model reports classification error (1 − accuracy). "
            "Full run completed: 2,114 models across 688 feature sets. "
            "Best AI Studio model: Fast Large Margin (accuracy ~43.5%). "
            "Logistic Regression error 0.885 (accuracy 11.5%) is treated as unreliable on this 37-class task."
        ),
    }


def plot_results(summary: dict, out_path: str):
    """Bar chart of classification error (%) for each completed AI Studio model."""
    completed = [m for m in summary["models"] if m["status"] == "completed"]
    if not completed:
        return

    names = [m["model"] for m in completed]
    errs = [m["classification_error"] * 100 for m in completed]
    stds = [(m["classification_error_std"] or 0) * 100 for m in completed]
    best = summary.get("best_model")
    colors = ["#2ca02c" if n == best else "#4c72b0" for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, errs, yerr=stds, capsize=4, color=colors, edgecolor="white")
    ax.set_ylabel("Classification error (%)")
    ax.set_title("AI Studio Auto Model — MIMIC sample (completed models)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    # Optional custom CSV path as first command-line argument
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"AI Studio export not found: {csv_path}\n"
            "Export Results_All.csv from AI Studio into data/, then re-run."
        )

    os.makedirs(FIG_DIR, exist_ok=True)

    summary = parse_results(csv_path)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_results(summary, os.path.join(FIG_DIR, "fig3_ai_studio_results.png"))

    print(f"Parsed: {csv_path}")
    print(f"Completed models: {summary['n_completed']} / {summary['n_total']}")
    print(f"Best AI Studio model: {summary['best_model']} "
          f"(accuracy ~ {summary['best_accuracy']:.1%})")
    print(f"Saved: {OUT_JSON}")
    print(f"Figure: {os.path.join(FIG_DIR, 'fig3_ai_studio_results.png')}")


if __name__ == "__main__":
    main()
