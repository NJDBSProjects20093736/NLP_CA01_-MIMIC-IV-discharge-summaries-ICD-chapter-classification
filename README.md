# MIMIC-IV ICD Chapter Classifier

**CA01 Task 2 — B9AI006 Natural Language Processing**  
Dublin Business School · MSc Artificial Intelligence · Student ID: 20093736

A supervised NLP pipeline that classifies MIMIC-IV discharge summaries into **37 ICD chapter labels**, following the **CRISP-DM** workflow and compared against an **Altair AI Studio Auto Model** benchmark.

---

## Overview

| Item | Detail |
|------|--------|
| **Task** | 37-class multiclass text classification |
| **Input** | De-identified discharge summary text |
| **Target** | ICD chapter label (e.g. `ICD10-I`, `ICD9-Respiratory`) |
| **Dataset** | 9,990 stratified records (270 per class) |
| **Best Python model** | TF-IDF + `LinearSVC` |
| **Best AI Studio model** | Fast Large Margin |
| **Advanced extension** | Bio_ClinicalBERT (Appendix A) — TF-IDF + SVM retained |

### Results

| Track | Best model | Accuracy | Weighted F1 | Notes |
|-------|------------|----------|-------------|-------|
| **Python (this repo)** | TF-IDF + SVM | **56.1%** | **0.554** | Selected for deployment |
| **AI Studio (MIMIC)** | Fast Large Margin | 43.5% | — | Required CA01 benchmark |
| **Moodle spreadsheet** | Naive Bayes | 81.2% | 0.796 | Reference only (different teaching dataset) |

---

## Project structure

```text
.
├── data/                              Processed datasets (included in repo)
│   ├── mimic_icd_sample.csv           Main modelling dataset (9,990 rows)
│   ├── mimic_icd_for_ai_studio_oneline_v2.tsv   AI Studio upload file
│   ├── Results_All.csv                Exported AI Studio Auto Model results
│   └── mimic_icd_sample.README.txt    Column documentation
├── models/
│   └── best_model.pkl                 Saved TF-IDF + LinearSVC pipeline
├── shared/                            Reusable preprocessing & path helpers
│   ├── paths.py
│   ├── text_preprocessing.py
│   └── representations.py
├── outputs/
│   ├── figures/                       Generated charts (EDA → deployment)
│   └── json/                          Metrics summaries
├── section2_data_understanding/       Dataset build & EDA
├── section3_data_preparation/         Preprocessing & representations
├── section4_modelling/                Classifier comparison & training
├── section5_evaluation/                 Hold-out evaluation & benchmarks
├── section6_deployment/               Workflow figure & prediction demo
├── Advanced Extension - Phase1/       Optional beyond-class BERT experiments
├── requirements.txt
└── README.md
```

---

## Data note

This repository includes the **processed sample dataset** (`data/mimic_icd_sample.csv`).  
**Raw MIMIC-IV files are not included** — they must be obtained separately from [PhysioNet](https://physionet.org/) under the MIMIC credentialed access agreement.

To rebuild the sample from raw files, place these in a local `raw_data/` folder and run Step 1:

- `discharge.csv` (MIMIC-IV-Note)
- `diagnoses_icd.csv`
- `d_icd_diagnoses.csv`

See `data/mimic_icd_sample.README.txt` for column definitions and sampling details.

---

## Setup

**Requirements:** Python 3.10+

```bash
# 1. Clone and enter the project
git clone https://github.com/NJDBSProjects20093736/NLP_CA01_-MIMIC-IV-discharge-summaries-ICD-chapter-classification.git
cd NLP_CA01_-MIMIC-IV-discharge-summaries-ICD-chapter-classification

# 2. Create a virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK resources (first run only)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## Quick start

If you only want to test the saved model:

```bash
python section6_deployment/run_predict_demo.py
```

**Needs:** `models/best_model.pkl`, `data/mimic_icd_sample.csv`  
**Produces:** terminal predictions + `outputs/figures/fig6_predict_demo_console.png`

---

## Full pipeline

Run all commands from the **project root**. Steps 2–8 use the included `mimic_icd_sample.csv` and do not require raw MIMIC files.

### Data Understanding

#### Step 1 — Build MIMIC ICD sample *(optional — raw files required)*

```bash
python section2_data_understanding/build_mimic_icd_sample.py
```

| | |
|---|---|
| **Needs** | Raw MIMIC-IV files in `raw_data/` (`discharge.csv`, `diagnoses_icd.csv`, `d_icd_diagnoses.csv`) |
| **Produces** | `data/mimic_icd_sample.csv`, `data/mimic_icd_for_ai_studio_oneline_v2.tsv`, `data/mimic_icd_sample.README.txt` |

#### Step 2 — Exploratory data analysis

```bash
python section2_data_understanding/run_eda.py
```

| | |
|---|---|
| **Needs** | `data/mimic_icd_sample.csv` |
| **Produces** | `outputs/figures/fig1_class_distribution.png`, `fig2_word_count_hist.png`, `fig3_word_count_by_class.png`, `data/class_counts.csv`, `data/kmeans_cluster_vs_label.csv` |

---

### Data Preparation

#### Step 3 — Text preprocessing

```bash
python section3_data_preparation/run_preprocessing.py
```

| | |
|---|---|
| **Needs** | `data/mimic_icd_sample.csv` |
| **Produces** | `outputs/json/preprocessing_summary.json`, `outputs/figures/fig4_preprocessing_pipeline.png`, `fig4_token_counts_before_after.png` |

#### Step 4 — Text representations

```bash
python section3_data_preparation/run_representations.py
```

| | |
|---|---|
| **Needs** | `data/mimic_icd_sample.csv` |
| **Produces** | `outputs/json/representations_summary.json`, `outputs/figures/fig5_matrix_shapes.png`, `fig5_sparsity.png` |

Compares **CountVectorizer**, **TF-IDF**, and **TF-IDF + TruncatedSVD**.

---

### Modelling

#### Step 5 — Train and compare classifiers

```bash
python section4_modelling/run_modelling.py
```

| | |
|---|---|
| **Needs** | `data/mimic_icd_sample.csv` |
| **Produces** | `models/best_model.pkl`, `outputs/json/modelling_summary.json`, `outputs/figures/fig6_cv_results.png`, `fig6_confusion_matrix.png` |

**Pipelines compared (10-fold stratified CV):**

| Pipeline | Representation | Classifier |
|----------|----------------|------------|
| Count + NB | CountVectorizer | Multinomial Naive Bayes |
| TF-IDF + NB | TF-IDF | Multinomial Naive Bayes |
| TF-IDF + LogReg | TF-IDF | Logistic Regression |
| **TF-IDF + SVM** | **TF-IDF** | **LinearSVC** ← best |
| TF-IDF + SVD + LogReg | TF-IDF + SVD | Logistic Regression |

---

### AI Studio benchmark

Altair AI Studio **Auto Model** is the required CA01 benchmark workflow.

1. Upload `data/mimic_icd_for_ai_studio_oneline_v2.tsv` to AI Studio.
2. Configure:
   - **Target:** `icd_chapter`
   - **Input:** `text`
   - **Text extraction:** enabled (1,000 features)
   - **Automatic feature selection:** enabled
   - **Models:** Naive Bayes, GLM, Logistic Regression, Fast Large Margin, Deep Learning
3. Export results as `data/Results_All.csv`.
4. Import into this project:

```bash
python section5_evaluation/import_ai_studio_results.py
```

| | |
|---|---|
| **Needs** | `data/Results_All.csv` |
| **Produces** | `outputs/json/ai_studio_results.json`, `outputs/figures/fig3_ai_studio_results.png` |

---

### Evaluation

#### Step 6 — Hold-out evaluation & benchmark comparison

```bash
python section5_evaluation/run_evaluation.py
```

| | |
|---|---|
| **Needs** | `data/mimic_icd_sample.csv`, `models/best_model.pkl`, `outputs/json/modelling_summary.json`, `outputs/json/ai_studio_results.json` |
| **Produces** | `outputs/json/evaluation_summary.json`, `outputs/figures/fig7_per_class_f1.png`, `fig7_benchmark_comparison.png`, `fig7_top_confusions.png` |

Includes 20% stratified hold-out evaluation, per-class F1, confusion analysis, and Python vs AI Studio comparison.

---

### Deployment

#### Step 7 — Deployment workflow figure

```bash
python section6_deployment/generate_deployment_workflow_figure.py
```

| | |
|---|---|
| **Produces** | `outputs/figures/fig6_deployment_workflow.png` |

#### Step 8 — Prediction demo

```bash
python section6_deployment/run_predict_demo.py
```

| | |
|---|---|
| **Needs** | `models/best_model.pkl`, `data/mimic_icd_sample.csv` |
| **Produces** | `outputs/figures/fig6_predict_demo_console.png` |

**Example usage:**

```python
import joblib

pipe = joblib.load("models/best_model.pkl")
text = "Patient admitted with chest pain and shortness of breath..."
label = pipe.predict([text])[0]
print("Suggested ICD chapter:", label)
```

---

## Advanced extension (Appendix A)

An optional beyond-class extension in `Advanced Extension - Phase1/` evaluates whether **Bio_ClinicalBERT** can improve ICD chapter prediction relative to the main TF-IDF + SVM pipeline. It is documented in **Appendix A** of the written report and does not replace `models/best_model.pkl`.

| Experiment | Description | Weighted F1 (BERT) | Weighted F1 (TF-IDF + SVM) |
|------------|-------------|--------------------|-----------------------------|
| Phase 1 — frozen embeddings | 999-note stratified subset | 0.210 | 0.349 |
| Fine-tuning — full dataset | 9,990 notes, same 80/20 split | 0.318 | 0.558 |

**Conclusion:** TF-IDF + `LinearSVC` remains the selected model. See `Advanced Extension - Phase1/README.md` for setup, scripts (`run_bert_poc.py`, `run_bert_finetune.py`), and full outputs.

---

## Workflow diagram

```mermaid
flowchart LR
    A[Discharge summary] --> B[TF-IDF + LinearSVC]
    B --> C[Suggested ICD chapter]
    C --> D[Clinical coder review]
    D --> E[Final ICD code]
```

The classifier is **decision-support only** — not a replacement for certified clinical coders.

---

## Key outputs

| File | Description |
|------|-------------|
| `models/best_model.pkl` | Fitted TF-IDF + LinearSVC pipeline |
| `outputs/json/modelling_summary.json` | Cross-validation results |
| `outputs/json/evaluation_summary.json` | Hold-out metrics & per-class F1 |
| `outputs/json/ai_studio_results.json` | AI Studio benchmark summary |
| `outputs/figures/fig7_benchmark_comparison.png` | Python vs AI Studio chart |
| `outputs/figures/fig6_deployment_workflow.png` | Human-in-the-loop workflow |
| `Advanced Extension - Phase1/outputs/figures/bert_finetune_comparison.png` | Appendix A — BERT vs TF-IDF comparison |

---

## Reproducibility

- `random_state = 42` used for sampling, CV, and hold-out splits
- **10-fold `StratifiedKFold`** for model selection
- **80/20 stratified hold-out** for detailed evaluation
- Preprocessing and vectorisation are wrapped in sklearn `Pipeline` objects to prevent data leakage

---

## Limitations

- Predicts **ICD chapters**, not full ICD-9/ICD-10 codes
- Balanced academic sample — does not reflect real hospital ICD prevalence
- Trained on de-identified US MIMIC-IV data; external validation required before clinical use
- TF-IDF + SVM may miss long-range clinical context captured by transformer models
- **Human review is mandatory** before any prediction influences official coding

---

## Technologies

Python · pandas · NumPy · scikit-learn · NLTK · matplotlib · seaborn · joblib · Altair AI Studio Auto Model

---

## References

- Johnson, A.E.W. et al. (2023). [MIMIC-IV](https://doi.org/10.1038/s41597-022-01899-x). *Scientific Data*.
- Johnson, A. et al. (2023). [MIMIC-IV-Note](https://doi.org/10.13026/1n74-ne17). PhysioNet.
- Pedregosa, F. et al. (2011). [scikit-learn](https://jmlr.org/papers/v12/pedregosa11a.html). *JMLR*.
- Dublin Business School (2026). B9AI006 NLP CA01 assessment brief & AutoModelling benchmark materials.
- Alsentzer, E. et al. (2019). [Publicly available clinical BERT embeddings](https://aclanthology.org/W19-1909/). *Clinical NLP Workshop*, pp. 72–78.

---

## Disclaimer

This project is for **academic use only**. It is not a certified clinical coding tool and must not be used for official diagnosis, billing, or healthcare decision-making without further validation, governance, and expert human review.

---

## Acknowledgments

Thanks to **Terri Hoare**, NLP Programme Coordinator at Dublin Business School, for setting the CA01 assessment, providing module guidance, and supporting access to the MIMIC-IV dataset used in this project.

---

## Author

**D Nadeesha Jayasuriya** — Student ID: 20093736  
Dublin Business School · MSc Artificial Intelligence
