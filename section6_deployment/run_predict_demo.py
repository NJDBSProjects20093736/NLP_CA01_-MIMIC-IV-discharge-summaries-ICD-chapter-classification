# -*- coding: utf-8 -*-
"""
Deployment — Load saved model and predict on sample discharge summaries.

NEEDS:  models/best_model.pkl
        data/mimic_icd_sample.csv

PRODUCES: console predictions (printed to terminal)
          outputs/figures/fig6_predict_demo_console.png (terminal screenshot)

Run:
    python section6_deployment/run_predict_demo.py
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SHARED = os.path.join(PROJECT_ROOT, "shared")
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

from paths import *  # noqa: E402,F403
ensure_dirs()

import joblib
import matplotlib.pyplot as plt
import pandas as pd

MODEL_PATH = MODEL_PKL
DATA = DATA_CSV
CONSOLE_PNG = os.path.join(FIGURES_DIR, "fig6_predict_demo_console.png")


def save_console_png(lines: list[str], out_path: str) -> None:
    """Render captured console text as a terminal-style PNG."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    text = "\n".join(lines)
    n_lines = max(len(lines), 1)
    fig_h = min(14, max(5, n_lines * 0.28))
    fig, ax = plt.subplots(figsize=(11, fig_h), facecolor="#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    ax.axis("off")
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        fontfamily="monospace",
        color="#d4d4d4",
    )
    ax.set_title(
        "Prediction demo — console output (run_predict_demo.py)",
        fontsize=11,
        fontweight="bold",
        color="#ffffff",
        loc="left",
        pad=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor="#1e1e1e", bbox_inches="tight")
    plt.close()


def main():
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    if not os.path.exists(MODEL_PATH):
        log("ERROR: Run section4_modelling/run_modelling.py first.")
        save_console_png(lines, CONSOLE_PNG)
        sys.exit(1)

    pipe = joblib.load(MODEL_PATH)
    log(f"Loaded: {MODEL_PATH}")
    log(f"Pipeline steps: {[name for name, _ in pipe.steps]}")
    log()

    # Three short synthetic examples + one real row from the dataset
    examples = [
        "Patient presents with acute chest pain, troponin elevated, diagnosed with myocardial infarction.",
        "Admission for major depressive episode, psychiatry consult, started on SSRI medication.",
        "Pregnant female at 38 weeks gestation, scheduled for caesarean section, routine prenatal care.",
    ]

    df = pd.read_csv(DATA)
    real_row = df.sample(1, random_state=42).iloc[0]
    examples.append(str(real_row["text"])[:500] + "...")

    labels_true = [None, None, None, real_row["label"]]

    for i, (text, true) in enumerate(zip(examples, labels_true), 1):
        pred = pipe.predict([text])[0]
        log(f"--- Example {i} ---")
        log(f"Text excerpt: {text[:120]}...")
        log(f"Predicted ICD chapter: {pred}")
        if true:
            log(f"Actual label (sample row): {true}")
            log(f"Match: {pred == true}")
        log()

    log("Demo complete — pipeline runs end-to-end.")

    save_console_png(lines, CONSOLE_PNG)
    print(f"Saved terminal screenshot: {CONSOLE_PNG}")


if __name__ == "__main__":
    main()
