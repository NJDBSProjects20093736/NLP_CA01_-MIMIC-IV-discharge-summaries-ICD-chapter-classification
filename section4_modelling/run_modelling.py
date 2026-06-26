
"""
Section 4 — Modelling, CV, best_model.pkl.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SHARED = os.path.join(PROJECT_ROOT, "shared")
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

from paths import *  
ensure_dirs()

import json
import os
import sys
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=UserWarning)

from representations import (
    RANDOM_STATE,
    SVD_COMPONENTS,
    make_count_vectorizer,
    make_tfidf_svd_pipeline,
    make_tfidf_vectorizer,
)

DATA = DATA_CSV
FIG_DIR = FIGURES_DIR
MODEL_DIR = MODELS_DIR
OUT_JSON = MODELLING_JSON
N_SPLITS = 10


def build_pipelines():
    """One sklearn Pipeline per representation + classifier (CA01 minimum set)."""
    logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    return {
        "Count + NB": Pipeline([
            ("vec", make_count_vectorizer()),
            ("clf", MultinomialNB()),
        ]),
        "TF-IDF + NB": Pipeline([
            ("vec", make_tfidf_vectorizer()),
            ("clf", MultinomialNB()),
        ]),
        "TF-IDF + LogReg": Pipeline([
            ("vec", make_tfidf_vectorizer()),
            ("clf", logreg),
        ]),
        "TF-IDF + SVM": Pipeline([
            ("vec", make_tfidf_vectorizer()),
            ("clf", LinearSVC(random_state=RANDOM_STATE)),
        ]),
        "TF-IDF+SVD + LogReg": Pipeline([
            ("tfidf", make_tfidf_vectorizer()),
            ("svd", TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
    }


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA)
    X = df["text"].astype(str)
    y = df["label"].astype(str)
    print(f"Documents: {len(df):,} | Classes: {y.nunique()}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]

    results = []
    pipelines = build_pipelines()

    for name, pipe in pipelines.items():
        print(f"\nCross-validating: {name} ({N_SPLITS}-fold)...")
        t0 = time.perf_counter()
        scores = cross_validate(
            pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise"
        )
        elapsed = time.perf_counter() - t0
        row = {
            "model": name,
            "accuracy": float(scores["test_accuracy"].mean()),
            "accuracy_std": float(scores["test_accuracy"].std()),
            "precision_weighted": float(scores["test_precision_weighted"].mean()),
            "recall_weighted": float(scores["test_recall_weighted"].mean()),
            "f1_weighted": float(scores["test_f1_weighted"].mean()),
            "f1_std": float(scores["test_f1_weighted"].std()),
            "cv_seconds": round(elapsed, 1),
        }
        results.append(row)
        print(f"  F1={row['f1_weighted']:.4f} (+/- {row['f1_std']:.4f})  time={elapsed:.0f}s")

    best = max(results, key=lambda r: r["f1_weighted"])
    best_name = best["model"]
    print(f"\nBest model by F1: {best_name}")

    # Train best pipeline on 80/20 hold-out for confusion matrix 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    best_pipe = pipelines[best_name]
    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)

    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best_pipe, model_path)
    print("Saved model:", model_path)

    # Confusion matrix (top 10 classes by frequency)
    top_labels = y_test.value_counts().head(10).index.tolist()
    mask = y_test.isin(top_labels)
    cm = confusion_matrix(y_test[mask], y_pred[mask], labels=top_labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=top_labels,
                yticklabels=top_labels, ax=ax, cbar_kws={"label": "Count"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {best_name} (top 10 ICD chapters, hold-out 20%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig6_confusion_matrix.png"), dpi=150)
    plt.close()

    # Bar chart of F1 scores 
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r["model"] for r in results]
    f1s = [r["f1_weighted"] for r in results]
    errs = [r["f1_std"] for r in results]
    colors = ["#4c72b0" if n != best_name else "#2ca02c" for n in names]
    ax.bar(names, f1s, yerr=errs, capsize=4, color=colors, edgecolor="white")
    ax.set_ylabel("Weighted F1")
    ax.set_ylim(0, max(f1s) * 1.15)
    ax.set_title(f"10-fold CV results (best: {best_name})")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig6_cv_results.png"), dpi=150)
    plt.close()

    report = classification_report(y_test, y_pred, zero_division=0)
    summary = {
        "n_splits": N_SPLITS,
        "n_documents": int(len(df)),
        "n_classes": int(y.nunique()),
        "model_results": results,
        "best_model": best_name,
        "holdout_classification_report": report,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", OUT_JSON)
    print("Figures:", FIG_DIR)


if __name__ == "__main__":
    main()
