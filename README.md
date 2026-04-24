# Model Comparison Pipeline (Production Script)

## Overview
This project implements a production-ready machine learning pipeline for comparing multiple classification models on the Petra Telecom churn dataset.  
The script supports command-line execution, logging, and reproducible results, moving from a notebook-style workflow to a structured CLI-based pipeline.

## Features
- Command-line interface using `argparse`
- Structured logging instead of print statements
- Data validation before training
- Support for `--dry-run` mode (no training)
- 5-fold stratified cross-validation
- Comparison of 6 model configurations:
  - Dummy (baseline)
  - Logistic Regression (default & balanced)
  - Decision Tree (depth=5)
  - Random Forest (default & balanced)
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - PR-AUC
- Automatic saving of results, plots, and best model

## Installation

Install dependencies:

bash
pip install -r requirements.txt




Log Output Explanation

The log output provides a detailed trace of the pipeline execution from start to finish:

Data Loading
The dataset was successfully loaded with 4500 rows and 14 columns, confirming that the input file was read correctly.
Data Validation
Validation passed, meaning:
All required features are present
The dataset is not empty
The target variable is binary
The churn distribution shows:
83.64% non-churn
16.36% churn
This confirms that the dataset is imbalanced, which is important for model selection and evaluation.
Train-Test Split
The data was split into:
3600 training samples
900 testing samples
Stratified sampling was used, so the churn rate (16.36%) is preserved in both sets.
Model Evaluation (Cross-Validation Phase)
Six models were evaluated using cross-validation:
Dummy (baseline)
Logistic Regression (default & balanced)
Decision Tree (depth=5)
Random Forest (default & balanced)
This phase computes performance metrics across folds to ensure robust comparison.
Final Model Training
Each model was then trained on the full training dataset.
This step prepares the models for final evaluation and saving.
Saving Outputs
The following outputs were successfully generated:
comparison_table.csv → contains performance metrics for all models
experiment_log.csv → records the experiment results with timestamp
best_model.joblib → stores the best-performing model
pr_curves.png → shows Precision-Recall curves for top models
calibration.png → shows probability calibration performance
Best Model Selection
The best model selected was:
Random Forest (default)
Selection was based on PR-AUC, which is suitable for imbalanced datasets.
Pipeline Completion
The final log confirms that the entire pipeline executed successfully without errors.
Key Insight from Logs

The logs confirm that:

The pipeline is fully functional and reproducible
Data imbalance was handled through model comparison
Tree-based models (Random Forest) performed best
All outputs were generated correctly and saved for further analysis




