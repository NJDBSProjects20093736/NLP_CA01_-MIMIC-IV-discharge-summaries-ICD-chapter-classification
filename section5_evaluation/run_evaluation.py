
"""
Evaluation — Hold-out metrics and benchmark comparison.

"""
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SHARED = os.path.join(PROJECT_ROOT, "shared")
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

from paths import * 
ensure_dirs()

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from representations import RANDOM_STATE

# Paths 
DATA = DATA_CSV                    # data/mimic_icd_sample.csv
MODEL_PATH = MODEL_PKL             # models/best_model.pkl
STEP6_JSON = MODELLING_JSON        # outputs/json/modelling_summary.json
FIG_DIR = FIGURES_DIR              # outputs/figures/
OUT_JSON = EVALUATION_JSON         # outputs/json/evaluation_summary.json

# Benmark
BENCHMARKS = {
    "Spreadsheet NB": {"accuracy": 0.812, "precision": 0.869, "recall": 0.739, "f1": 0.796},
    "Spreadsheet GLM": {"accuracy": 0.638, "precision": 0.602, "recall": 0.821, "f1": 0.693},
    "Spreadsheet Deep Learning": {"accuracy": 0.750, "precision": 0.711, "recall": 0.831, "f1": 0.766},
}


def load_ai_studio_results() -> dict:
    """Load completed AI Studio models from outputs/json/ai_studio_results.json."""
    if not os.path.exists(AI_STUDIO_JSON):
        return {}
    with open(AI_STUDIO_JSON, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for m in data.get("models", []):
        if m.get("status") != "completed":
            continue
        # LR with error > 0.8 is unreliable on this 37-class task
        if m["model"] == "Logistic Regression" and m["classification_error"] > 0.8:
            continue
        out[m["model"]] = {
            "accuracy": m["accuracy"],
            "classification_error": m["classification_error"],
            "f1": None,
        }
    return out


def per_class_from_report(y_true, y_pred) -> list[dict]:
    """Structured per-class metrics from sklearn classification_report."""
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    rows = []
    for label, stats in report.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        rows.append({
            "label": label,
            "precision": float(stats["precision"]),
            "recall": float(stats["recall"]),
            "f1": float(stats["f1-score"]),
            "support": int(stats["support"]),
        })
    return rows


def top_confusion_pairs(y_true, y_pred, labels, n=12):
    """Return the most frequent off-diagonal confusion pairs."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pairs = []
    for i, true_lbl in enumerate(labels):
        for j, pred_lbl in enumerate(labels):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                pairs.append({
                    "true": true_lbl,
                    "predicted": pred_lbl,
                    "count": count,
                })
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:n]


def plot_per_class_f1(per_class: list[dict], out_path: str):
    """Horizontal bar chart of F1 per ICD chapter."""
    df = pd.DataFrame(per_class).sort_values("f1", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 10))
    colors = ["#c44e52" if f < 0.40 else "#4c72b0" if f < 0.60 else "#2ca02c" for f in df["f1"]]
    ax.barh(df["label"], df["f1"], color=colors, edgecolor="white")
    ax.axvline(0.56, color="gray", linestyle="--", linewidth=1, label="Weighted avg ≈ 0.56")
    ax.set_xlabel("F1 score (hold-out 20%)")
    ax.set_title("Per-class F1 — best model (TF-IDF + SVM)")
    ax.set_xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_benchmark_comparison(
    best_cv_f1: float,
    best_holdout_f1: float,
    best_cv_accuracy: float,
    ai_studio: dict,
    out_path: str,
):
    """Bar chart: spreadsheet F1, Python F1, AI Studio accuracy (MIMIC)."""
    names = list(BENCHMARKS.keys())
    f1s = [BENCHMARKS[k]["f1"] for k in BENCHMARKS]
    colors = ["#aaaaaa"] * len(BENCHMARKS)

    names += ["Python TF-IDF+SVM (CV F1)", "Python TF-IDF+SVM"]
    f1s += [best_cv_f1, best_holdout_f1]
    colors += ["#4c72b0", "#2ca02c"]

    for model_name, metrics in ai_studio.items():
        short = model_name.replace("Generalized Linear Model", "GLM")
        names.append(f"AI Studio {short} (acc.)")
        f1s.append(metrics["accuracy"])
        colors.append("#dd8452")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(names, f1s, color=colors, edgecolor="white")
    ax.set_ylabel("F1 (spreadsheet / Python) or Accuracy (AI Studio)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Benchmark comparison")
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
                ha="center", va="bottom", fontsize=7)
    plt.xticks(rotation=28, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_top_confusions(pairs: list[dict], out_path: str):
    """Bar chart of most common misclassification pairs."""
    if not pairs:
        return
    labels = [f"{p['true']} → {p['predicted']}" for p in pairs]
    counts = [p["count"] for p in pairs]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels[::-1], counts[::-1], color="#dd8452", edgecolor="white")
    ax.set_xlabel("Count (hold-out set)")
    ax.set_title("Top misclassification pairs — best model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # CV metrics and best model 
    with open(STEP6_JSON, encoding="utf-8") as f:
        step6 = json.load(f)

    best_name = step6["best_model"]
    best_cv = next(r for r in step6["model_results"] if r["model"] == best_name)
    ai_studio = load_ai_studio_results()  # empty dict if import script not run yet

    df = pd.read_csv(DATA)
    X = df["text"].astype(str)
    y = df["label"].astype(str)

    # Same hold-out split as run_modelling.py (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipe = joblib.load(MODEL_PATH)
    y_pred = pipe.predict(X_test)

    report_text = classification_report(y_test, y_pred, zero_division=0)
    per_class = per_class_from_report(y_test, y_pred)

    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    holdout_weighted = {
        "accuracy": float(report_dict["accuracy"]),
        "precision_weighted": float(report_dict["weighted avg"]["precision"]),
        "recall_weighted": float(report_dict["weighted avg"]["recall"]),
        "f1_weighted": float(report_dict["weighted avg"]["f1-score"]),
    }

    labels = sorted(y_test.unique())
    confusions = top_confusion_pairs(y_test, y_pred, labels, n=12)

    sorted_by_f1 = sorted(per_class, key=lambda r: r["f1"])
    weakest = sorted_by_f1[:5]
    strongest = sorted_by_f1[-5:][::-1]

    # figures
    plot_per_class_f1(per_class, os.path.join(FIG_DIR, "fig7_per_class_f1.png"))
    plot_benchmark_comparison(
        best_cv["f1_weighted"],
        holdout_weighted["f1_weighted"],
        best_cv["accuracy"],
        ai_studio,
        os.path.join(FIG_DIR, "fig7_benchmark_comparison.png"),
    )
    plot_top_confusions(confusions, os.path.join(FIG_DIR, "fig7_top_confusions.png"))

    summary = {
        "best_model": best_name,
        "holdout_split": "80/20 stratified, random_state=42",
        "n_test": int(len(y_test)),
        "cv_metrics": {
            "accuracy": best_cv["accuracy"],
            "precision_weighted": best_cv["precision_weighted"],
            "recall_weighted": best_cv["recall_weighted"],
            "f1_weighted": best_cv["f1_weighted"],
            "f1_std": best_cv["f1_std"],
        },
        "holdout_metrics": holdout_weighted,
        "per_class": per_class,
        "strongest_classes": strongest,
        "weakest_classes": weakest,
        "top_confusions": confusions,
        "benchmarks_spreadsheet": BENCHMARKS,
        "ai_studio_mimic": ai_studio or None,
        "classification_report": report_text,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Best model: {best_name}")
    print(f"Hold-out F1 (weighted): {holdout_weighted['f1_weighted']:.4f}")
    print(f"Saved: {OUT_JSON}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
