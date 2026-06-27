# -*- coding: utf-8 -*-
"""
Path constants for the Advanced submission package.

"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
JSON_DIR = os.path.join(OUTPUTS_DIR, "json")
CACHE_DIR = os.path.join(OUTPUTS_DIR, "cache")

_CANDIDATE_DATA_PATHS = [
    os.path.join(ROOT, "..", "data", "mimic_icd_sample.csv"),
    os.path.join(ROOT, "..", "..", "Submission", "data", "mimic_icd_sample.csv"),
]

_MAIN_EVAL_JSON_PATHS = [
    os.path.join(ROOT, "..", "outputs", "json", "evaluation_summary.json"),
    os.path.join(ROOT, "..", "..", "Submission", "outputs", "json", "evaluation_summary.json"),
]

POC_SUMMARY_JSON = os.path.join(JSON_DIR, "bert_poc_summary.json")
POC_FIGURE = os.path.join(FIGURES_DIR, "bert_poc_comparison.png")
SAMPLE_CSV = os.path.join(CACHE_DIR, "poc_stratified_sample.csv")
EMB_TRAIN_NPY = os.path.join(CACHE_DIR, "bert_embeddings_train.npy")
EMB_TEST_NPY = os.path.join(CACHE_DIR, "bert_embeddings_test.npy")

FINETUNE_SUMMARY_JSON = os.path.join(JSON_DIR, "bert_finetune_summary.json")
FINETUNE_FIGURE = os.path.join(FIGURES_DIR, "bert_finetune_comparison.png")
FINETUNE_MODEL_DIR = os.path.join(CACHE_DIR, "bert_finetune_model")
SPLIT_JSON = os.path.join(CACHE_DIR, "finetune_split_indices.json")


def _first_existing(candidates: list[str]) -> str | None:
    seen = set()
    for path in candidates:
        path = os.path.abspath(path)
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            return path
    return None


def resolve_data_csv() -> str:
    """Return path to mimic_icd_sample.csv from the main CA01 project."""
    path = _first_existing(_CANDIDATE_DATA_PATHS)
    if path:
        return path
    raise FileNotFoundError(
        "mimic_icd_sample.csv not found. Expected at ../data/ (GitHub repo) "
        "or ../../Submission/data/ (local CA01 folder)."
    )


def resolve_main_eval_json() -> str | None:
    """Return path to main project evaluation_summary.json if available."""
    return _first_existing(_MAIN_EVAL_JSON_PATHS)


def ensure_dirs() -> None:
    """Create output and cache folders if they do not exist."""
    for d in (OUTPUTS_DIR, FIGURES_DIR, JSON_DIR, CACHE_DIR, FINETUNE_MODEL_DIR):
        os.makedirs(d, exist_ok=True)
