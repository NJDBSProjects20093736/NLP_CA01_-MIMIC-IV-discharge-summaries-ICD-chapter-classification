
"""
Section 3.6 — Text representations.
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

import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse

from representations import REPRESENTATIONS, MAX_FEATURES, NGRAM_RANGE, SVD_COMPONENTS

DATA = DATA_CSV
FIG_DIR = FIGURES_DIR
OUT_JSON = REPRESENTATIONS_JSON


def matrix_stats(name, X, elapsed):
    """Summarise feature matrix for the report."""
    if sparse.issparse(X):
        n_rows, n_cols = X.shape
        sparsity = 1.0 - (X.nnz / (n_rows * n_cols))
        density_pct = (1.0 - sparsity) * 100
    else:
        n_rows, n_cols = X.shape
        sparsity = None
        density_pct = 100.0
    return {
        "name": name,
        "shape_rows": int(n_rows),
        "shape_features": int(n_cols),
        "nonzero_pct": round(density_pct, 4),
        "sparsity_pct": round(sparsity * 100, 2) if sparsity is not None else 0.0,
        "fit_transform_seconds": round(elapsed, 2),
    }


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("Loading", DATA)
    df = pd.read_csv(DATA)
    texts = df["text"].astype(str).tolist()
    print(f"Documents: {len(texts):,}")

    results = []
    shapes = []
    sparsities = []

    for name, factory in REPRESENTATIONS.items():
        print(f"Fitting: {name} ...")
        t0 = time.perf_counter()
        model = factory()
        X = model.fit_transform(texts)
        elapsed = time.perf_counter() - t0
        stats = matrix_stats(name, X, elapsed)
        results.append(stats)
        shapes.append((name, stats["shape_features"]))
        sparsities.append((name, stats["sparsity_pct"]))
        print(f"  shape={stats['shape_rows']} x {stats['shape_features']}  "
              f"sparsity={stats['sparsity_pct']:.2f}%  time={elapsed:.1f}s")

    summary = {
        "n_documents": len(texts),
        "ngram_range": list(NGRAM_RANGE),
        "max_features": MAX_FEATURES,
        "svd_components": SVD_COMPONENTS,
        "representations": results,
        "sample_features": {},
    }

    # Top 10 TF-IDF terms from one ICD10-I example 
    from representations import make_tfidf_vectorizer
    tfidf = make_tfidf_vectorizer()
    X_ex = tfidf.fit_transform(texts[:500])
    idx = list(tfidf.get_feature_names_out()[:10])
    summary["sample_features"]["tfidf_vocab_head"] = idx

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Figure: feature dimensions
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [s[0].replace(" + n-grams", "").replace(" (BoW)", "") for s in shapes]
    vals = [s[1] for s in shapes]
    colors = ["steelblue", "teal", "darkorange"]
    ax.bar(names, vals, color=colors, edgecolor="white")
    ax.set_ylabel("Number of features")
    ax.set_title("Feature matrix width after each representation (n = 9,990 docs)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_matrix_shapes.png"), dpi=150)
    plt.close()

    # Figure: sparsity
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, [s[1] for s in sparsities], color=colors, edgecolor="white")
    ax.set_ylabel("Sparsity (%)")
    ax.set_title("Matrix sparsity (higher = more empty cells)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_sparsity.png"), dpi=150)
    plt.close()

    print("Saved:", OUT_JSON)
    print("Figures:", FIG_DIR)


if __name__ == "__main__":
    main()
