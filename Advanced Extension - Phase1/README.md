# Advanced Extension

This folder documents a beyond-class extension to the MIMIC ICD chapter classifier.
It evaluates whether a clinical transformer model (Bio_ClinicalBERT) can improve ICD
chapter prediction relative to the main TF-IDF + Linear SVM pipeline developed in
the primary CA01 submission.

The selected model for the main CA01 project remains **TF-IDF + LinearSVC**
(weighted F1 = 0.554 on the full dataset, 10-fold stratified cross-validation).

## 1. Project Overview

The primary project applied TF-IDF with a Linear Support Vector Machine and
compared performance against the required Altair AI Studio Auto Model benchmark.
This extension investigates two complementary transformer-based approaches:

* **Phase 1 (proof of concept):** frozen Bio_ClinicalBERT embeddings on a
  stratified subset of 999 discharge summaries, with a logistic regression
  classifier on top of the embedding vectors
* **Fine-tuning experiment:** end-to-end fine-tuning of Bio_ClinicalBERT on the
  full dataset of 9,990 discharge summaries across 37 ICD chapter classes

Both experiments were conducted independently of the main scikit-learn pipeline.
Neither approach outperformed the TF-IDF + SVM baseline on the evaluation metrics
reported below. The classical pipeline was therefore retained as the final selected
model for evaluation and deployment.

## 2. Setup

### Requirements

* Python 3.10 or higher
* CUDA-capable GPU recommended for embedding extraction and fine-tuning
* Processed dataset
* Python packages listed in `requirements-advanced.txt`
* Git LFS for version control of the fine-tuned model weights (~413 MB)

### Installation

Dependencies are installed from this folder:

```bash
pip install -r requirements-advanced.txt
git lfs install
```

NLTK resources are not required for the advanced scripts.

### Data dependency

The scripts consume the processed modelling dataset produced by the main CA01
pipeline:

```text
../data/mimic_icd_sample.csv
```

Path resolution is handled in `shared/paths.py`, which also supports a local
layout at `../../Submission/data/mimic_icd_sample.csv`. Raw MIMIC-IV source files
are not required; only the processed CSV from the primary submission is used.

---

## 3. Project Structure

```text
shared/                        Path helpers, BERT embedding utilities, PyTorch dataset
outputs/figures/               Comparison charts
outputs/json/                  Metrics summaries (pre-generated)
outputs/cache/                 Embeddings, stratified sample, fine-tuned model weights

run_bert_poc.py                Phase 1 — frozen embeddings proof of concept
run_bert_finetune.py           Fine-tune Bio_ClinicalBERT on the full dataset
requirements-advanced.txt      torch, transformers, accelerate, scikit-learn
README.txt
```

### Repository contents

| Component | Description |
|-----------|-------------|
| `run_bert_poc.py` | Phase 1 experiment script |
| `run_bert_finetune.py` | Full-dataset fine-tuning experiment |
| `shared/` | Shared helper modules (`paths`, `clinical_embeddings`, `bert_dataset`) |
| `outputs/json/` | JSON summaries of evaluation metrics |
| `outputs/figures/` | Bar charts comparing TF-IDF + SVM against BERT approaches |
| `outputs/cache/` | Intermediate and final artefacts (embeddings, sample CSV, model weights) |
| `.gitattributes` | Git LFS configuration for `model.safetensors` |
| `.gitignore` | Exclusions for Python cache files and logs |

The fine-tuned weights file (`outputs/cache/bert_finetune_model/model.safetensors`,
approximately 413 MB) exceeds GitHub's standard file size limit and is tracked
via Git LFS. The total size of this folder is approximately 427 MB.

---

## 4. Pipeline Description

### 4.1 Phase 1 — Frozen BERT Embeddings (Proof of Concept)

```bash
python run_bert_poc.py
```

**Input:**

```text
../data/mimic_icd_sample.csv
```

**Output:**

```text
outputs/json/bert_poc_summary.json
outputs/figures/bert_poc_comparison.png
outputs/cache/poc_stratified_sample.csv
outputs/cache/bert_embeddings_train.npy
outputs/cache/bert_embeddings_test.npy
```

**Method:**

A stratified sample of 999 notes (27 per ICD chapter class) is drawn from the main
dataset. TF-IDF + LinearSVC is compared against frozen Bio_ClinicalBERT embeddings
with a logistic regression head. Both models use the same 80/20 stratified
hold-out split (`random_state = 42`). Typical runtime on GPU is approximately
three minutes.

### 4.2 Fine-tuning — Bio_ClinicalBERT (Full Dataset)

```bash
python run_bert_finetune.py
```

**Input:**

```text
../data/mimic_icd_sample.csv
```

**Output:**

```text
outputs/json/bert_finetune_summary.json
outputs/figures/bert_finetune_comparison.png
outputs/cache/finetune_split_indices.json
outputs/cache/bert_finetune_model/
```

**Method:**

All 9,990 discharge summaries and 37 ICD chapter labels are used. The model
`emilyalsentzer/Bio_ClinicalBERT` is fine-tuned with a 37-class classification
head and compared against TF-IDF + LinearSVC on the same 80/20 stratified
hold-out split. Typical runtime on GPU is approximately 30–40 minutes.

**Hyperparameters used:**

| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-5 |
| Batch size | 16 |
| Epochs | 3 (with early stopping) |
| Max sequence length | 512 tokens |
| Random seed | 42 |

Optional overrides are available via environment variables: `FT_EPOCHS`,
`FT_BATCH_SIZE`, `FT_LR`, and `FT_MAX_LENGTH`.

## 5. Results

The following results were generated during the extension work and are stored in
`outputs/json/`.

### Fine-tuning experiment (full dataset, 80/20 hold-out split)

| Model | Accuracy | Weighted F1 |
|-------|----------|-------------|
| TF-IDF + LinearSVC (same split) | 56.3% | 0.558 |
| Fine-tuned Bio_ClinicalBERT | 36.2% | 0.318 |
| TF-IDF + SVM (main CA01, 10-fold CV) | 56.1% | 0.554 |

### Phase 1 proof of concept (999-note subset)

| Model | Weighted F1 |
|-------|-------------|
| TF-IDF + LinearSVC (subset) | 0.349 |
| Bio_ClinicalBERT + Logistic Regression (subset) | 0.210 |

**Conclusion:** TF-IDF + LinearSVC was retained as the selected model for the main
CA01 pipeline, evaluation, and deployment. The fine-tuning comparison chart
(`outputs/figures/bert_finetune_comparison.png`) is referenced as Figure A.1 in
Appendix A of the written report.

## 6. Reproducibility

* `random_state = 42` is used for train/test splitting and model training where
  applicable
* The fine-tuning experiment uses an 80/20 stratified hold-out split on the full
  dataset
* Phase 1 draws 27 notes per class (999 total) from the same source CSV
* Pre-trained weights: `emilyalsentzer/Bio_ClinicalBERT` (Hugging Face)
