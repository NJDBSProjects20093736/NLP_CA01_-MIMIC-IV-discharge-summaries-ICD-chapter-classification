MIMIC ICD Chapter Classifier

A Natural Language Processing project for predicting ICD chapter labels from MIMIC-IV discharge summaries using supervised machine learning.

This project follows the CRISP-DM workflow:

Data Understanding → Data Preparation → Modelling → Evaluation → Deployment

The final Python model is a scikit-learn pipeline using TF-IDF + Linear SVM, compared against an Altair AI Studio Auto Model benchmark.

Project Overview

This project builds an ICD chapter classifier using discharge summary text from MIMIC-IV-Note linked with diagnosis information from MIMIC-IV.

The task is a 37-class multiclass text classification problem, where each discharge summary is mapped to one ICD chapter label.

The project includes:

MIMIC-IV dataset construction
Exploratory data analysis
Text preprocessing
CountVectorizer, TF-IDF, and TF-IDF + SVD representations
Supervised classifier comparison
10-fold stratified cross-validation
Altair AI Studio Auto Model benchmark
Hold-out evaluation and confusion analysis
Human-in-the-loop deployment demo
Optional beyond-class Bio_ClinicalBERT extension (Appendix A)
Final Model

The best Python model was:

TF-IDF + LinearSVC

Expected performance:

Accuracy: approximately 56%
Weighted F1: approximately 0.55

The best AI Studio benchmark model was:

Fast Large Margin
Accuracy: approximately 43.5%
Classification error: approximately 56.5%
Project Structure
data/                          Input datasets and processed CSV/TSV files
models/                        Saved trained model pipeline
shared/                        Shared helper code
outputs/figures/               Generated figures and charts
outputs/json/                  Generated metrics and summaries

section2_data_understanding/   Dataset building and exploratory data analysis
section3_data_preparation/     Text preprocessing and feature representation
section4_modelling/            Classifier comparison and model training
section5_evaluation/           Evaluation metrics, confusion analysis, and benchmarks
section6_deployment/           Prediction demo and deployment workflow
Advanced Extension - Phase1/   Optional beyond-class BERT experiments (Appendix A)
Setup
1. Create and activate a Python environment
python -m venv .venv

Windows:

.venv\Scripts\activate

macOS/Linux:

source .venv/bin/activate
2. Install dependencies
pip install -r requirements.txt
3. Download NLTK resources
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
How to Run

Run all scripts from the project root folder.

Step 1 — Build the MIMIC ICD Sample
python section2_data_understanding/build_mimic_icd_sample.py

This script builds the final project dataset from raw MIMIC-IV files.

Required raw files:

discharge.csv
diagnoses_icd.csv
d_icd_diagnoses.csv

Outputs:

data/mimic_icd_sample.csv
data/mimic_icd_for_ai_studio_oneline_v2.tsv
data/mimic_icd_sample.README.txt

The TSV file is used for the Altair AI Studio Auto Model benchmark.

Step 2 — Run Exploratory Data Analysis
python section2_data_understanding/run_eda.py

Outputs:

outputs/figures/fig1_class_distribution.png
outputs/figures/fig2_word_count_hist.png
outputs/figures/fig3_word_count_by_class.png
data/class_counts.csv
data/kmeans_cluster_vs_label.csv

This step explores the dataset size, class distribution, vocabulary, word counts, and sample discharge summaries.

Step 3 — Run Text Preprocessing
python section3_data_preparation/run_preprocessing.py

Outputs:

outputs/json/preprocessing_summary.json
outputs/figures/fig4_preprocessing_pipeline.png
outputs/figures/fig4_token_counts_before_after.png

This step demonstrates the text cleaning and preprocessing workflow.

Step 4 — Generate Text Representations
python section3_data_preparation/run_representations.py

Outputs:

outputs/json/representations_summary.json
outputs/figures/fig5_matrix_shapes.png
outputs/figures/fig5_sparsity.png

This step compares different text representations:

CountVectorizer
TF-IDF
TF-IDF + TruncatedSVD
Step 5 — Train and Compare Models
python section4_modelling/run_modelling.py

Outputs:

models/best_model.pkl
outputs/json/modelling_summary.json
outputs/figures/fig6_cv_results.png
outputs/figures/fig6_confusion_matrix.png

The following pipelines are compared:

Pipeline	Text Representation	Classifier
Count + NB	CountVectorizer	Multinomial Naive Bayes
TF-IDF + NB	TF-IDF	Multinomial Naive Bayes
TF-IDF + Logistic Regression	TF-IDF	Logistic Regression
TF-IDF + SVM	TF-IDF	LinearSVC
TF-IDF + SVD + Logistic Regression	TF-IDF + TruncatedSVD	Logistic Regression

The selected model is saved as:

models/best_model.pkl
AI Studio Auto Model Benchmark

Altair AI Studio Auto Model is used as the required benchmark workflow.

Upload this file to AI Studio:

data/mimic_icd_for_ai_studio_oneline_v2.tsv

Recommended AI Studio setup:

Target: icd_chapter
Input: text
Task: Classification
Text extraction: enabled
Extracted text features: 1,000
Automatic feature selection: enabled

Enabled model types:

Naive Bayes
Generalized Linear Model
Logistic Regression
Fast Large Margin
Deep Learning

After running Auto Model, export the results as:

data/Results_All.csv

Then import the AI Studio results into the Python project:

python section5_evaluation/import_ai_studio_results.py

Outputs:

outputs/json/ai_studio_results.json
outputs/figures/fig3_ai_studio_results.png
Evaluation

Run the evaluation script:

python section5_evaluation/run_evaluation.py

Outputs:

outputs/json/evaluation_summary.json
outputs/figures/fig7_per_class_f1.png
outputs/figures/fig7_benchmark_comparison.png
outputs/figures/fig7_top_confusions.png

The evaluation includes:

20% stratified hold-out test evaluation
Classification report
Per-class F1 analysis
Confusion matrix
Frequent misclassification pairs
Benchmark comparison with AI Studio Auto Model
Deployment Demo
Generate Human-in-the-Loop Workflow Figure
python section6_deployment/generate_deployment_workflow_figure.py

Output:

outputs/figures/fig6_deployment_workflow.png
Run Prediction Demo
python section6_deployment/run_predict_demo.py

Outputs:

outputs/figures/fig6_predict_demo_console.png

The demo loads the saved model and predicts an ICD chapter for sample discharge summary text.

Example prediction structure:

import joblib

pipe = joblib.load("models/best_model.pkl")

text = "Patient admitted with chest pain and shortness of breath..."
label = pipe.predict([text])[0]

print("Suggested ICD chapter:", label)
Advanced Extension (Appendix A)

An optional beyond-class extension is provided in Advanced Extension - Phase1/.
It evaluates whether Bio_ClinicalBERT can improve ICD chapter prediction relative
to the main TF-IDF + LinearSVC pipeline. This work is reported in Appendix A and
does not replace models/best_model.pkl.

Two experiments were run:

Phase 1 (frozen embeddings, 999-note subset):
  TF-IDF + LinearSVC weighted F1: 0.349
  Bio_ClinicalBERT + LogReg weighted F1: 0.210

Fine-tuning (full 9,990 notes, same 80/20 split):
  TF-IDF + LinearSVC weighted F1: 0.558
  Fine-tuned Bio_ClinicalBERT weighted F1: 0.318

TF-IDF + LinearSVC was retained as the selected model. See Advanced Extension - Phase1/README.txt for scripts, setup, and outputs.

Human-in-the-Loop Use

This classifier is designed as a decision-support tool, not a replacement for certified clinical coders or clinicians.

In a real workflow:

A discharge summary is exported from the EHR or coding queue.
The saved model predicts a suggested ICD chapter.
A clinical coder reviews the prediction.
The final ICD code is assigned by the human coder.
Low-confidence or weak-category predictions are flagged for mandatory review.

The model predicts broad ICD chapter labels only. It does not predict complete ICD-9 or ICD-10 codes.

Limitations

Key limitations include:

The model predicts ICD chapters, not full ICD codes.
The dataset is a balanced academic sample, not a natural hospital distribution.
The model was trained on de-identified MIMIC-IV data from a US hospital setting.
External validation would be required before real clinical deployment.
TF-IDF + SVM may miss deeper clinical context and long-range relationships.
Human review is required before any prediction could influence official coding.
Key Outputs

Important generated files:

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
Advanced Extension - Phase1/outputs/figures/bert_finetune_comparison.png
Reproducibility

The project uses fixed random seeds where applicable:

random_state = 42

The main Python model comparison uses:

10-fold StratifiedKFold cross-validation
shuffle = True
random_state = 42
Technologies Used
Python
pandas
NumPy
scikit-learn
NLTK
matplotlib
joblib
Altair AI Studio Auto Model
Bio_ClinicalBERT (Alsentzer et al., 2019) — advanced extension only
References
MIMIC-IV
MIMIC-IV-Note
scikit-learn
Altair AI Studio Auto Model
DBS B9AI006 NLP CA01 assessment materials
Disclaimer

This project is for academic use only. It is not a certified clinical coding tool and should not be used for official diagnosis, billing, or healthcare decision-making without further validation, governance, and expert human review.
