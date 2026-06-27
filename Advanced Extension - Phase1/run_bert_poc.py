
"""
Frozen Bio_ClinicalBERT embeddings on a 999-note stratified subset, compared
against TF-IDF + LinearSVC on the same 80/20 hold-out split.

"""
from __future__ import annotations

import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Allow imports from shared/ when run from project root
_SHARED = os.path.join(os.path.dirname(__file__), "shared")
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

from clinical_embeddings import ClinicalEmbedder  
from paths import (  
    EMB_TEST_NPY,
    EMB_TRAIN_NPY,
    resolve_main_eval_json,
    POC_FIGURE,
    POC_SUMMARY_JSON,
    SAMPLE_CSV,
    ensure_dirs,
    resolve_data_csv,
)

RANDOM_STATE = 42
LABEL_COL = "label"
TEXT_COL = "text"


def stratified_subset(df: pd.DataFrame, n_per_class: int) -> pd.DataFrame:
    """Sample n_per_class rows from each ICD chapter label."""
    parts = []
    for label, group in df.groupby(LABEL_COL):
        n = min(n_per_class, len(group))
        parts.append(group.sample(n=n, random_state=RANDOM_STATE))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def metric_dict(y_true, y_pred) -> dict:
    """Return accuracy, weighted precision/recall/F1 for multiclass evaluation."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def run_tfidf_baseline(x_train, x_test, y_train, y_test) -> dict:
    """TF-IDF + LinearSVC baseline on the same POC train/test split."""
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=10_000,
                ),
            ),
            ("clf", LinearSVC(random_state=RANDOM_STATE)),
        ]
    )
    t0 = time.perf_counter()
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    elapsed = time.perf_counter() - t0
    return {"metrics": metric_dict(y_test, y_pred), "train_seconds": round(elapsed, 2)}


def run_bert_logreg(x_train, x_test, y_train, y_test, batch_size: int) -> dict:
    """Frozen Bio_ClinicalBERT embeddings + Logistic Regression classifier."""
    embedder = ClinicalEmbedder()
    t0 = time.perf_counter()

    # Reuse cached embeddings on reruns (saves GPU time)
    if os.path.isfile(EMB_TRAIN_NPY) and os.path.isfile(EMB_TEST_NPY):
        print("Loading cached BERT embeddings...")
        x_train_emb = np.load(EMB_TRAIN_NPY)
        x_test_emb = np.load(EMB_TEST_NPY)
        embed_seconds = 0.0
    else:
        print(f"BERT device: {embedder.device}")
        x_train_emb = embedder.embed_batch(x_train, batch_size=batch_size)
        x_test_emb = embedder.embed_batch(x_test, batch_size=batch_size)
        np.save(EMB_TRAIN_NPY, x_train_emb)
        np.save(EMB_TEST_NPY, x_test_emb)
        embed_seconds = time.perf_counter() - t0

    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    t1 = time.perf_counter()
    clf.fit(x_train_emb, y_train)
    y_pred = clf.predict(x_test_emb)
    train_seconds = time.perf_counter() - t1

    return {
        "metrics": metric_dict(y_test, y_pred),
        "model": "Bio_ClinicalBERT mean-pool + LogisticRegression",
        "bert_model": embedder.model_name,
        "max_length": 512,
        "embedding_dim": int(x_train_emb.shape[1]),
        "embed_seconds": round(embed_seconds, 2),
        "classifier_train_seconds": round(train_seconds, 2),
    }


def load_main_project_reference() -> dict | None:
    """Load main CA01 evaluation metrics for comparison chart"""
    path = resolve_main_eval_json()
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "model": data.get("best_model", "TF-IDF + SVM"),
        "scope": "full dataset (9,990 notes)",
        "cv_metrics": data.get("cv_metrics"),
        "holdout_metrics": data.get("holdout_metrics"),
    }


def plot_comparison(results: dict, out_path: str) -> None:
    """Bar chart: POC subset F1 vs main project reference."""
    rows = [
        ("TF-IDF + SVM (POC subset)", results["tfidf_baseline"]["metrics"]["f1_weighted"]),
        ("BioClinicalBERT + LogReg (POC subset)", results["bert_logreg"]["metrics"]["f1_weighted"]),
    ]
    ref = results.get("main_project_reference")
    if ref and ref.get("holdout_metrics"):
        rows.append(
            (
                f"{ref['model']} (main project, full data)",
                ref["holdout_metrics"]["f1_weighted"],
            )
        )

    labels, scores = zip(*rows)
    plt.figure(figsize=(9, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52"][: len(rows)]
    bars = plt.bar(labels, scores, color=colors)
    plt.ylim(0, max(scores) * 1.25)
    plt.ylabel("Weighted F1")
    plt.title("Phase 1 POC: BERT vs TF-IDF (stratified subset)")
    plt.xticks(rotation=15, ha="right")
    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center",
        )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    ensure_dirs()
    n_per_class = int(os.environ.get("POC_N_PER_CLASS", "27"))
    batch_size = int(os.environ.get("POC_BATCH_SIZE", "16"))
    skip_bert = os.environ.get("POC_SKIP_BERT", "") == "1"

    # Load main dataset and build stratified POC subset
    data_path = resolve_data_csv()
    print(f"Data: {data_path}")

    df = pd.read_csv(data_path)
    subset = stratified_subset(df, n_per_class=n_per_class)
    subset.to_csv(SAMPLE_CSV, index=False)
    print(f"POC subset: {len(subset)} rows, {subset[LABEL_COL].nunique()} classes")

    texts = subset[TEXT_COL].astype(str).tolist()
    labels = subset[LABEL_COL].astype(str).tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    print(f"Train: {len(x_train)} | Test: {len(x_test)}")

    # Baseline: same split, TF-IDF + LinearSVC 
    print("\n--- TF-IDF + LinearSVC baseline (same POC subset) ---")
    tfidf_result = run_tfidf_baseline(x_train, x_test, y_train, y_test)
    print(json.dumps(tfidf_result["metrics"], indent=2))

    # Advanced: frozen BERT embeddings + Logistic Regression
    bert_result = None
    if not skip_bert:
        print("\n--- Bio_ClinicalBERT + LogisticRegression (POC subset) ---")
        bert_result = run_bert_logreg(x_train, x_test, y_train, y_test, batch_size=batch_size)
        print(json.dumps(bert_result["metrics"], indent=2))

    summary = {
        "phase": "Advanced Phase 1 - proof of concept",
        "note": (
            "Separate experiment; main CA01 pipeline unchanged. "
            "POC metrics are on a stratified subset only."
        ),
        "data_source": data_path,
        "subset_csv": SAMPLE_CSV,
        "n_per_class": n_per_class,
        "n_subset": len(subset),
        "n_classes": int(subset[LABEL_COL].nunique()),
        "split": "80/20 stratified hold-out, random_state=42",
        "n_train": len(x_train),
        "n_test": len(x_test),
        "tfidf_baseline": {
            "model": "TF-IDF + LinearSVC",
            "metrics": tfidf_result["metrics"],
            "train_seconds": tfidf_result["train_seconds"],
        },
        "bert_logreg": bert_result,
        "main_project_reference": load_main_project_reference(),
    }

    with open(POC_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if bert_result:
        plot_comparison(summary, POC_FIGURE)

    print(f"\nSaved: {POC_SUMMARY_JSON}")
    if bert_result:
        print(f"Saved: {POC_FIGURE}")


if __name__ == "__main__":
    main()
