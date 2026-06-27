
"""
Fine-tune Bio_ClinicalBERT on full MIMIC ICD sample.

"""
from __future__ import annotations

import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

_SHARED = os.path.join(os.path.dirname(__file__), "shared")
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

from bert_dataset import ClinicalTextDataset  
from clinical_embeddings import DEFAULT_MODEL  
from paths import ( 
    FINETUNE_FIGURE,
    FINETUNE_MODEL_DIR,
    FINETUNE_SUMMARY_JSON,
    resolve_main_eval_json,
    SPLIT_JSON,
    ensure_dirs,
    resolve_data_csv,
)

RANDOM_STATE = 42
LABEL_COL = "label"
TEXT_COL = "text"


def metric_dict(y_true, y_pred) -> dict:
    """Return accuracy and weighted precision/recall/F1."""
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


def load_main_project_reference() -> dict | None:
    """Load main CA01 CV/hold-out metrics for the comparison chart."""
    path = resolve_main_eval_json()
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "model": data.get("best_model", "TF-IDF + SVM"),
        "scope": "full dataset, 10-fold CV (main CA01)",
        "cv_metrics": data.get("cv_metrics"),
        "holdout_metrics": data.get("holdout_metrics"),
    }


def run_tfidf_same_split(x_train, x_test, y_train, y_test) -> dict:
    """TF-IDF + LinearSVC on the identical train/test split used for BERT."""
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
    return {
        "model": "TF-IDF + LinearSVC (same 80/20 split)",
        "metrics": metric_dict(y_test, y_pred),
        "train_seconds": round(time.perf_counter() - t0, 2),
    }


def compute_metrics(eval_pred):
    """Hugging Face Trainer callback: weighted F1 and accuracy per epoch."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


def plot_comparison(summary: dict, out_path: str) -> None:
    """Bar chart for Appendix A Figure A.1."""
    rows = [
        (
            "TF-IDF + SVM\n(same hold-out split)",
            summary["tfidf_same_split"]["metrics"]["f1_weighted"],
        ),
        (
            "Fine-tuned Bio_ClinicalBERT\n(same hold-out split)",
            summary["bert_finetune"]["metrics"]["f1_weighted"],
        ),
    ]
    ref = summary.get("main_project_reference")
    if ref and ref.get("cv_metrics"):
        rows.append(
            (
                f"{ref['model']}\n(main project CV)",
                ref["cv_metrics"]["f1_weighted"],
            )
        )

    labels, scores = zip(*rows)
    colors = ["#4C72B0", "#55A868", "#C44E52"][: len(rows)]
    plt.figure(figsize=(9, 5.5))
    bars = plt.bar(labels, scores, color=colors)
    plt.ylim(0, max(scores) * 1.2)
    plt.ylabel("Weighted F1")
    plt.title("Fine-tuned Bio_ClinicalBERT vs TF-IDF (full dataset)")
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
    epochs = int(os.environ.get("FT_EPOCHS", "3"))
    batch_size = int(os.environ.get("FT_BATCH_SIZE", "16"))
    learning_rate = float(os.environ.get("FT_LR", "2e-5"))
    max_length = int(os.environ.get("FT_MAX_LENGTH", "512"))

    # Load full dataset and create stratified 80/20 split 
    data_path = resolve_data_csv()
    print(f"Data: {data_path}")
    df = pd.read_csv(data_path)
    texts = df[TEXT_COL].astype(str).tolist()
    raw_labels = df[LABEL_COL].astype(str).tolist()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(raw_labels)
    n_classes = len(label_encoder.classes_)
    print(f"Rows: {len(df)} | Classes: {n_classes}")

    x_train, x_test, y_train, y_test, y_train_str, y_test_str = train_test_split(
        texts,
        y,
        raw_labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    with open(SPLIT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "random_state": RANDOM_STATE,
                "n_train": len(x_train),
                "n_test": len(x_test),
                "label_classes": label_encoder.classes_.tolist(),
            },
            f,
            indent=2,
        )

    print(f"Train: {len(x_train)} | Test: {len(x_test)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Fair baseline on the same split
    print("\n--- TF-IDF + LinearSVC on same split ---")
    tfidf_result = run_tfidf_same_split(x_train, x_test, y_train_str, y_test_str)
    print(json.dumps(tfidf_result["metrics"], indent=2))

    # Fine-tune Bio_ClinicalBERT with 37-class head
    print("\n--- Fine-tuning Bio_ClinicalBERT ---")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        DEFAULT_MODEL,
        num_labels=n_classes,
        id2label={i: str(l) for i, l in enumerate(label_encoder.classes_)},
        label2id={str(l): i for i, l in enumerate(label_encoder.classes_)},
    )

    train_ds = ClinicalTextDataset(x_train, y_train.tolist(), tokenizer, max_length=max_length)
    test_ds = ClinicalTextDataset(x_test, y_test.tolist(), tokenizer, max_length=max_length)

    training_args = TrainingArguments(
        output_dir=os.path.join(FINETUNE_MODEL_DIR, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        logging_steps=100,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    t0 = time.perf_counter()
    train_result = trainer.train()
    train_seconds = round(time.perf_counter() - t0, 2)

    # Hold-out evaluation and save artefacts
    print("\n--- Evaluating fine-tuned model on hold-out test set ---")
    pred = trainer.predict(test_ds)
    y_pred = np.argmax(pred.predictions, axis=-1)
    bert_metrics = metric_dict(y_test, y_pred)
    print(json.dumps(bert_metrics, indent=2))

    trainer.save_model(FINETUNE_MODEL_DIR)
    tokenizer.save_pretrained(FINETUNE_MODEL_DIR)

    y_pred_labels = label_encoder.inverse_transform(y_pred)
    report = classification_report(
        y_test_str,
        y_pred_labels,
        zero_division=0,
        output_dict=True,
    )

    summary = {
        "phase": "Advanced Appendix A - Bio_ClinicalBERT fine-tuning",
        "note": (
            "Beyond-class extension on full 9,990-note sample. "
            "Same 80/20 stratified split for TF-IDF and BERT comparison."
        ),
        "data_source": data_path,
        "n_rows": len(df),
        "n_classes": n_classes,
        "split": "80/20 stratified hold-out",
        "random_state": RANDOM_STATE,
        "n_train": len(x_train),
        "n_test": len(x_test),
        "bert_model": DEFAULT_MODEL,
        "max_length": max_length,
        "training": {
            "epochs_requested": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_seconds": train_seconds,
            "train_loss": train_result.training_loss,
            "device": device,
        },
        "tfidf_same_split": tfidf_result,
        "bert_finetune": {
            "model": "Fine-tuned Bio_ClinicalBERT",
            "metrics": bert_metrics,
            "saved_model_dir": FINETUNE_MODEL_DIR,
        },
        "classification_report": report,
        "main_project_reference": load_main_project_reference(),
        "winner_same_split": (
            "Bio_ClinicalBERT"
            if bert_metrics["f1_weighted"] > tfidf_result["metrics"]["f1_weighted"]
            else "TF-IDF + LinearSVC"
        ),
    }

    with open(FINETUNE_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_comparison(summary, FINETUNE_FIGURE)
    print(f"\nWinner (same hold-out): {summary['winner_same_split']}")
    print(f"Saved: {FINETUNE_SUMMARY_JSON}")
    print(f"Saved: {FINETUNE_FIGURE}")
    print(f"Saved model: {FINETUNE_MODEL_DIR}")


if __name__ == "__main__":
    main()
