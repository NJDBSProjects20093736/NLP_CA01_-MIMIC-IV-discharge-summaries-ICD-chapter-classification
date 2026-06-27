
"""Shared path constants for the MIMIC ICD chapter classifier project."""
import os

# Project root = parent of this shared/ folder
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
JSON_DIR = os.path.join(OUTPUTS_DIR, "json")

DATA_CSV = os.path.join(DATA_DIR, "mimic_icd_sample.csv")
DATA_README = os.path.join(DATA_DIR, "mimic_icd_sample.README.txt")
AI_STUDIO_TSV = os.path.join(DATA_DIR, "mimic_icd_for_ai_studio_oneline_v2.tsv")
CLASS_COUNTS_CSV = os.path.join(DATA_DIR, "class_counts.csv")
KMEANS_CSV = os.path.join(DATA_DIR, "kmeans_cluster_vs_label.csv")

MODEL_PKL = os.path.join(MODELS_DIR, "best_model.pkl")
AI_STUDIO_CSV = os.path.join(DATA_DIR, "Results_All.csv")
AI_STUDIO_JSON = os.path.join(JSON_DIR, "ai_studio_results.json")
MODELLING_JSON = os.path.join(JSON_DIR, "modelling_summary.json")
PREPROCESS_JSON = os.path.join(JSON_DIR, "preprocessing_summary.json")
REPRESENTATIONS_JSON = os.path.join(JSON_DIR, "representations_summary.json")
EVALUATION_JSON = os.path.join(JSON_DIR, "evaluation_summary.json")


def ensure_dirs():
    """Create data, models, and output folders if they do not exist."""
    for d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, JSON_DIR):
        os.makedirs(d, exist_ok=True)
