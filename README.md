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

```bash
pip install -r requirements.txt
