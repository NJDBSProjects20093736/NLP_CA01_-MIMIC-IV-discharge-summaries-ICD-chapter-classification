# CA01 Task 2 — MIMIC ICD Chapter Classifier

This project builds a supervised NLP classifier for predicting ICD chapter labels from MIMIC-IV discharge summaries. It follows the CRISP-DM workflow:

**Data Understanding → Data Preparation → Modelling → Evaluation → Deployment**

The final model is a Python `scikit-learn` pipeline using **TF-IDF + Linear SVM**, compared against an **Altair AI Studio Auto Model** benchmark.

---

## 1. Project Overview

The aim of this project is to classify discharge summary text into ICD chapter labels. The project uses discharge summaries from MIMIC-IV-Note and diagnosis information from MIMIC-IV.

The project includes:

* Dataset construction from MIMIC-IV files
* Exploratory data analysis
* Text preprocessing
* Text representation using CountVectorizer, TF-IDF, and TF-IDF + SVD
* Classifier comparison using 10-fold stratified cross-validation
* AI Studio Auto Model benchmark comparison
* Hold-out evaluation and confusion analysis
* Human-in-the-loop deployment demonstration

---

## 2. Setup Instructions

### Requirements

* Python 3.10 or higher
* Required Python libraries listed in `requirements.txt`

### Installation

From the project root folder, run:

```bash
pip install -r requirements.txt
```

Download the required NLTK resources:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## 3. Project Structure

```text
data/                          Input datasets and processed CSV/TSV files
models/                        Saved trained model pipeline
shared/                        Shared helper code for paths, preprocessing, and representations
outputs/figures/               Generated figures and charts
outputs/json/                  Generated metrics and summaries

section2_data_understanding/   Dataset building and exploratory data analysis
section3_data_preparation/     Text preprocessing and feature representation
section4_modelling/            Classifier comparison and model training
section5_evaluation/           Evaluation metrics, confusion analysis, and benchmarks
section6_deployment/           Prediction demo and deployment workflow
```

---

## 4. How to Run the Project

Run the scripts from the project root folder in the order below.

---

## 4.1 Data Understanding

### Step 1: Build the MIMIC ICD Sample

This step creates the final project dataset from the raw MIMIC-IV files.

```bash
python section2_data_understanding/build_mimic_icd_sample.py
```

**Needs:**

Raw MIMIC-IV files in the raw data folder:

```text
discharge.csv
diagnoses_icd.csv
d_icd_diagnoses.csv
```

**Produces:**

```text
data/mimic_icd_sample.csv
data/mimic_icd_for_ai_studio_oneline_v2.tsv
data/mimic_icd_sample.README.txt
```

The CSV file contains the main modelling dataset. The TSV file is formatted for upload to Altair AI Studio.

---

### Step 2: Run Exploratory Data Analysis

```bash
python section2_data_understanding/run_eda.py
```

**Needs:**

```text
data/mimic_icd_sample.csv
```

**Produces:**

```text
outputs/figures/fig1_class_distribution.png
outputs/figures/fig2_word_count_hist.png
outputs/figures/fig3_word_count_by_class.png
data/class_counts.csv
data/kmeans_cluster_vs_label.csv
```

This step also prints summary information to the console, including row counts, number of classes, vocabulary information, and sample text excerpts.

---

## 4.2 Data Preparation

### Step 3: Run Text Preprocessing

```bash
python section3_data_preparation/run_preprocessing.py
```

**Needs:**

```text
data/mimic_icd_sample.csv
```

**Produces:**

```text
outputs/json/preprocessing_summary.json
outputs/figures/fig4_preprocessing_pipeline.png
outputs/figures/fig4_token_counts_before_after.png
```

This step demonstrates the preprocessing process used for the discharge summary text.

---

### Step 4: Generate Text Representations

```bash
python section3_data_preparation/run_representations.py
```

**Needs:**

```text
data/mimic_icd_sample.csv
```

**Produces:**

```text
outputs/json/representations_summary.json
outputs/figures/fig5_matrix_shapes.png
outputs/figures/fig5_sparsity.png
```

This step creates and compares CountVectorizer, TF-IDF, and TF-IDF + SVD representations.

---

## 4.3 Modelling

### Step 5: Train and Compare Classifiers

```bash
python section4_modelling/run_modelling.py
```

**Needs:**

```text
data/mimic_icd_sample.csv
```

**Produces:**

```text
models/best_model.pkl
outputs/json/modelling_summary.json
outputs/figures/fig6_cv_results.png
outputs/figures/fig6_confusion_matrix.png
```

This step compares the Python modelling pipelines using 10-fold stratified cross-validation.

The tested pipelines include:

* Count + Naive Bayes
* TF-IDF + Naive Bayes
* TF-IDF + Logistic Regression
* TF-IDF + Linear SVM
* TF-IDF + SVD + Logistic Regression

The expected best model is:

```text
TF-IDF + LinearSVC
```

---

## 4.4 AI Studio Benchmark

Altair AI Studio Auto Model is used as the required benchmark workflow.

### AI Studio Input File

Upload the following file to AI Studio:

```text
data/mimic_icd_for_ai_studio_oneline_v2.tsv
```

### AI Studio Configuration

Use:

```text
Target: icd_chapter
Input: text
Task: Classification
Text extraction: enabled
Extracted text features: 1,000
Automatic feature selection: enabled
```

Enabled model types:

* Naive Bayes
* Generalized Linear Model
* Logistic Regression
* Fast Large Margin
* Deep Learning

After running Auto Model, export the results file as:

```text
data/Results_All.csv
```

Then import the AI Studio results into the Python project:

```bash
python section5_evaluation/import_ai_studio_results.py
```

**Needs:**

```text
data/Results_All.csv
```

**Produces:**

```text
outputs/json/ai_studio_results.json
outputs/figures/fig3_ai_studio_results.png
```

---

## 4.5 Evaluation

### Step 6: Run Hold-out Evaluation and Benchmark Comparison

```bash
python section5_evaluation/run_evaluation.py
```

**Needs:**

```text
data/mimic_icd_sample.csv
models/best_model.pkl
outputs/json/modelling_summary.json
outputs/json/ai_studio_results.json
```

**Produces:**

```text
outputs/json/evaluation_summary.json
outputs/figures/fig7_per_class_f1.png
outputs/figures/fig7_benchmark_comparison.png
outputs/figures/fig7_top_confusions.png
```

This step evaluates the selected TF-IDF + SVM model on a 20% stratified hold-out test set. It also compares the Python model against the AI Studio Auto Model benchmark.

---

## 4.6 Deployment

### Step 7: Generate Deployment Workflow Figure

```bash
python section6_deployment/generate_deployment_workflow_figure.py
```

**Produces:**

```text
outputs/figures/fig6_deployment_workflow.png
```

This figure shows the proposed human-in-the-loop deployment workflow.

---

### Step 8: Run Demo Prediction Script

```bash
python section6_deployment/run_predict_demo.py
```

**Needs:**

```text
models/best_model.pkl
data/mimic_icd_sample.csv
```

**Produces:**

```text
outputs/figures/fig6_predict_demo_console.png
```

This script loads the saved model and predicts ICD chapter labels for sample discharge summaries.

---

## 5. Expected Results

The expected best Python model is:

```text
TF-IDF + LinearSVC
```

Typical performance:

```text
Accuracy: approximately 56%
Weighted F1: approximately 0.55
```

The best AI Studio Auto Model benchmark is expected to be:

```text
Fast Large Margin
Accuracy: approximately 43.5%
Classification error: approximately 56.5%
```

---

## 6. Key Output Files

Important generated outputs include:

```text
models/best_model.pkl
outputs/json/modelling_summary.json
outputs/json/evaluation_summary.json
outputs/json/ai_studio_results.json
outputs/figures/fig6_cv_results.png
outputs/figures/fig7_per_class_f1.png
outputs/figures/fig7_benchmark_comparison.png
outputs/figures/fig7_top_confusions.png
outputs/figures/fig6_deployment_workflow.png
outputs/figures/fig6_predict_demo_console.png
```

---

## 7. Notes on Deployment

The saved model is intended for demonstration and decision-support use only. It predicts broad ICD chapter labels, not full ICD-9 or ICD-10 codes. In a real clinical coding workflow, predictions should always be reviewed by a qualified clinical coder before official use.

The model should not be used as a fully automated clinical coding system without further validation, external testing, monitoring, and governance.

---

## 8. Reproducibility Notes

The project uses fixed random seeds where applicable, including:

```text
random_state = 42
```

The Python modelling workflow uses 10-fold stratified cross-validation to support fair model comparison across all ICD chapter classes.
