
"""Data Understanding EDA
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SHARED = os.path.join(PROJECT_ROOT, "shared")
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

from paths import *  
ensure_dirs()

import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Paths

DATA = DATA_CSV
# Columns: text, label, primary_icd_code, primary_icd_title, note_id, hadm_id
# Refer data/mimic_icd_sample.README.txt for full column documentation.
FIG_DIR = FIGURES_DIR
CLASS_COUNTS_CSV = CLASS_COUNTS_CSV
KMEANS_CSV = KMEANS_CSV

os.makedirs(FIG_DIR, exist_ok=True)
sns.set_style("whitegrid")


def token_estimate(text: str) -> int:
    """
    Approximate word count using regex word boundaries.

    """
    return len(re.findall(r"\b\w+\b", str(text).lower()))


def print_dataset_overview(df: pd.DataFrame) -> None:
   
    print("\n=== DATASET OVERVIEW ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"Classes: {df['label'].nunique()}")
    print(f"Missing text: {df['text'].isna().sum()}")
    print(f"Duplicate note_id: {df['note_id'].duplicated().sum()}")


def plot_class_distribution(df: pd.DataFrame) -> pd.Series:
    """
    balanced ICD chapter distribution.
    Returns value counts sorted by label name.
    """
    print("\n=== CLASS DISTRIBUTION  ===")
    counts = df["label"].value_counts().sort_index()
    print(counts.to_string())
    counts.to_csv(CLASS_COUNTS_CSV)
    print(f"Saved: {CLASS_COUNTS_CSV}")

    plt.figure(figsize=(14, 6))
    counts.plot(kind="bar", color="steelblue", edgecolor="white")
    plt.title("ICD Chapter Class Distribution (MIMIC-IV Sample, n=9,990)")
    plt.xlabel("ICD chapter label")
    plt.ylabel("Number of discharge summaries")
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_class_distribution.png"), dpi=150)
    plt.close()
    return counts


def plot_document_lengths(df: pd.DataFrame, counts: pd.Series) -> None:
    """word-count distribution overall and by sample of chapters."""
    print("\n=== DOCUMENT LENGTH  ===")
    desc = df["word_count"].describe()
    print(desc)
    print(f"Mean: {df['word_count'].mean():.0f} | Median: {df['word_count'].median():.0f}")

    # Histogram 
    plt.figure(figsize=(8, 4))
    plt.hist(df["word_count"], bins=50, color="teal", edgecolor="white")
    plt.title("Figure 2: Discharge Summary Length Distribution")
    plt.xlabel("Word count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_word_count_hist.png"), dpi=150)
    plt.close()

    # Boxplot 
    top8 = counts.head(8).index
    sub = df[df["label"].isin(top8)]
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=sub, x="label", y="word_count", order=list(top8))
    plt.title("Figure 3: Word Count by ICD Chapter (sample of 8 classes)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_word_count_by_class.png"), dpi=150)
    plt.close()


def print_vocabulary_estimate(df: pd.DataFrame) -> None:
    """ rough corpus vocabulary size ."""
    print("\n=== VOCABULARY ESTIMATE (Section 2.7) ===")
    all_words = " ".join(df["text"].astype(str).str.lower()).split()
    vocab = len(set(all_words))
    print(f"Total tokens (approx): {len(all_words):,}")
    print(f"Unique tokens (approx): {vocab:,}")


def print_example_documents(df: pd.DataFrame, labels: list[str]) -> None:
    """short excerpts for the report."""
    print("\n=== SAMPLE TEXTS (Section 2.8) ===")
    for label in labels:
        subset = df[df["label"] == label]
        if subset.empty:
            print(f"\n--- {label}: no rows found ---")
            continue
        row = subset.iloc[0]
        print(f"\n--- {label} | {row['primary_icd_code']} | {row['primary_icd_title']} ---")
        print(str(row["text"])[:600], "...\n")


def optional_kmeans_eda(df: pd.DataFrame, n_sample: int = 3000, k: int = 10) -> None:
    """
    unsupervised clusters align ICD chapters.

    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        print("\n=== OPTIONAL K-MEANS EDA  ===")
        sample = df.sample(n=min(n_sample, len(df)), random_state=42)
        vec = TfidfVectorizer(max_features=5000, stop_words="english")
        X = vec.fit_transform(sample["text"])
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        sample = sample.copy()
        sample["cluster"] = km.fit_predict(X)
        ctab = pd.crosstab(sample["cluster"], sample["label"])
        ctab.to_csv(KMEANS_CSV)
        print(f"Saved cluster vs label crosstab: {KMEANS_CSV}")
        print("(Clusters do not map 1:1 to labels — expected for unsupervised EDA.)")
    except Exception as exc:
        print(f"K-means EDA skipped: {exc}")


def main():
    if not os.path.exists(DATA):
        raise FileNotFoundError(
            f"{DATA} not found. Run: python section2_data_understanding/build_mimic_icd_sample.py"
        )

    print("Loading", DATA)
    df = pd.read_csv(DATA)

    # Derived length features for EDA 
    df["word_count"] = df["text"].astype(str).apply(token_estimate)
    df["char_count"] = df["text"].astype(str).str.len()

    print_dataset_overview(df)
    counts = plot_class_distribution(df)
    plot_document_lengths(df, counts)

    print("\n=== ICD VERSION MIX (primary code first character) ===")
    print(df["primary_icd_code"].astype(str).str[0].value_counts().head())

    print_vocabulary_estimate(df)
    print_example_documents(df, ["ICD10-I", "ICD9-Respiratory", "ICD10-C"])
    optional_kmeans_eda(df)

    print(f"\nFigures saved to: {FIG_DIR}")
    print("Section 2 EDA complete.")


if __name__ == "__main__":
    main()
