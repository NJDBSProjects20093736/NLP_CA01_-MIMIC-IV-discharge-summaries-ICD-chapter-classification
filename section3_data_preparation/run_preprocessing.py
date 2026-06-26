
"""
Section 3 — Preprocessing.
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

import matplotlib.pyplot as plt
import pandas as pd

from text_preprocessing import preprocess_text, normalize_whitespace

DATA = DATA_CSV
FIG_DIR = FIGURES_DIR
OUT_JSON = PREPROCESS_JSON


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(DATA)

    # Build before/after example
    raw = str(df.iloc[0]["text"])
    excerpt = normalize_whitespace(raw)[:1200]
    steps = {
        "raw_excerpt": excerpt,
        "after_lowercase_punct": preprocess_text(excerpt, lemmatize=False, remove_stopwords=False),
        "after_stopwords": preprocess_text(excerpt, lemmatize=False, remove_stopwords=True),
        "after_lemmatize": preprocess_text(excerpt, lemmatize=True, remove_stopwords=True),
    }

    # Apply full preprocessing to entire corpus 
    print("Preprocessing 9,990 documents...")
    df["text_preprocessed"] = df["text"].astype(str).map(
        lambda t: preprocess_text(t, lemmatize=True, remove_stopwords=True)
    )

    raw_lens = df["text"].astype(str).str.split().map(len)
    prep_lens = df["text_preprocessed"].str.split().map(len)

    summary = {
        "n_documents": int(len(df)),
        "raw_tokens_mean": float(raw_lens.mean()),
        "preprocessed_tokens_mean": float(prep_lens.mean()),
        "raw_tokens_median": float(raw_lens.median()),
        "preprocessed_tokens_median": float(prep_lens.median()),
        "example_label": str(df.iloc[0]["label"]),
        "preprocessing_steps": steps,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Figure: token count distribution 
    sub = df.sample(n=500, random_state=42)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sub["text"].astype(str).str.split().map(len), bins=30, alpha=0.6, label="Raw", color="steelblue")
    ax.hist(sub["text_preprocessed"].str.split().map(len), bins=30, alpha=0.6, label="Preprocessed", color="darkorange")
    ax.set_xlabel("Tokens per document")
    ax.set_ylabel("Frequency")
    ax.set_title("Token counts before vs after preprocessing (n=500 sample)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_token_counts_before_after.png"), dpi=150)
    plt.close()

    # Figure: pipeline flowchart
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        "Raw discharge\ntext",
        "Normalise\nwhitespace",
        "Lowercase +\nremove punct",
        "Tokenise\n(NLTK)",
        "Remove stop\nwords",
        "Lemmatise\n(WordNet)",
    ]
    n = len(stages)
    xs = [0.06 + i * (0.88 / (n - 1)) for i in range(n)]
    box_w, box_h = 0.11, 0.42
    for x, s in zip(xs, stages):
        ax.text(
            x, 0.5, s, ha="center", va="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#e8f4ff", edgecolor="#333"),
        )
    for i in range(n - 1):
        ax.annotate(
            "",
            xy=(xs[i + 1] - box_w / 2 - 0.01, 0.5),
            xytext=(xs[i] + box_w / 2 + 0.01, 0.5),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="#333"),
        )
    ax.text(
        0.5, 0.08,
        "Output: space-separated preprocessed string  →  CountVectorizer / TF-IDF (Section 3.6)",
        ha="center", va="center", fontsize=8, style="italic", color="#444",
    )
    ax.set_title(
        "Text preprocessing pipeline (code/text_preprocessing.py)",
        fontsize=11, fontweight="bold", pad=14,
    )
    plt.savefig(
        os.path.join(FIG_DIR, "fig4_preprocessing_pipeline.png"),
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    print(f"Mean tokens: {summary['raw_tokens_mean']:.0f} -> {summary['preprocessed_tokens_mean']:.0f}")
    print("Saved:", OUT_JSON)
    print("Figures:", FIG_DIR)


if __name__ == "__main__":
    main()
