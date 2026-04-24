"""
Module 5 Week B — Stretch: From Notebook to Production Script

Run:
    python compare_models.py --data-path data/telecom_churn.csv
    python compare_models.py --data-path data/telecom_churn.csv --dry-run
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibrationDisplay
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
    "senior_citizen",
    "has_partner",
    "has_dependents",
    "contract_months",
]

TARGET_COLUMN = "churned"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Petra Telecom churn model comparison pipeline."
    )
    parser.add_argument("--data-path", required=True, help="Path to input CSV dataset.")
    parser.add_argument("--output-dir", default="./output", help="Directory for saved outputs.")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and show configuration without training models.",
    )
    return parser.parse_args()


def load_data(data_path):
    if not os.path.exists(data_path):
        logging.error("Data file not found: %s", data_path)
        sys.exit(1)

    df = pd.read_csv(data_path)
    logging.info("Loaded data: %s rows, %s columns", df.shape[0], df.shape[1])
    return df


def validate_data(df):
    required_columns = NUMERIC_FEATURES + [TARGET_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        logging.error("Missing required columns: %s", missing)
        sys.exit(1)

    if df.empty:
        logging.error("Dataset is empty.")
        sys.exit(1)

    if df[TARGET_COLUMN].nunique() != 2:
        logging.error("Target column must be binary.")
        sys.exit(1)

    logging.info("Data validation passed.")
    logging.info("Dataset shape: %s", df.shape)
    logging.info(
        "Target distribution:\n%s",
        df[TARGET_COLUMN].value_counts(normalize=True).to_string(),
    )


def define_models(random_seed):
    return {
        "Dummy": Pipeline([
            ("scaler", "passthrough"),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "LR_default": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=random_seed)),
        ]),
        "LR_balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_seed,
            )),
        ]),
        "DT_depth5": Pipeline([
            ("scaler", "passthrough"),
            ("model", DecisionTreeClassifier(max_depth=5, random_state=random_seed)),
        ]),
        "RF_default": Pipeline([
            ("scaler", "passthrough"),
            ("model", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_seed,
            )),
        ]),
        "RF_balanced": Pipeline([
            ("scaler", "passthrough"),
            ("model", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=random_seed,
            )),
        ]),
    }


def run_cv_comparison(models, X, y, n_folds, random_seed):
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    results = []

    for name, model in models.items():
        logging.info("Evaluating model: %s", name)

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        pr_auc_scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]

            accuracy_scores.append(accuracy_score(y_val, y_pred))
            precision_scores.append(precision_score(y_val, y_pred, zero_division=0))
            recall_scores.append(recall_score(y_val, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_val, y_pred, zero_division=0))
            pr_auc_scores.append(average_precision_score(y_val, y_proba))

        results.append({
            "model": name,
            "accuracy_mean": np.mean(accuracy_scores),
            "accuracy_std": np.std(accuracy_scores),
            "precision_mean": np.mean(precision_scores),
            "precision_std": np.std(precision_scores),
            "recall_mean": np.mean(recall_scores),
            "recall_std": np.std(recall_scores),
            "f1_mean": np.mean(f1_scores),
            "f1_std": np.std(f1_scores),
            "pr_auc_mean": np.mean(pr_auc_scores),
            "pr_auc_std": np.std(pr_auc_scores),
        })

    return pd.DataFrame(results)


def fit_models(models, X_train, y_train):
    fitted_models = {}

    for name, model in models.items():
        logging.info("Fitting final model: %s", name)
        fitted_models[name] = model.fit(X_train, y_train)

    return fitted_models


def get_top3_models_by_pr_auc(fitted_models, X_test, y_test):
    scores = {}

    for name, model in fitted_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        scores[name] = average_precision_score(y_test, y_proba)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]


def save_outputs(results_df, fitted_models, X_test, y_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    comparison_path = os.path.join(output_dir, "comparison_table.csv")
    results_df.to_csv(comparison_path, index=False)
    logging.info("Saved comparison table to %s", comparison_path)

    log_df = pd.DataFrame({
        "model_name": results_df["model"],
        "accuracy": results_df["accuracy_mean"],
        "precision": results_df["precision_mean"],
        "recall": results_df["recall_mean"],
        "f1": results_df["f1_mean"],
        "pr_auc": results_df["pr_auc_mean"],
        "timestamp": datetime.now().isoformat(),
    })

    log_path = os.path.join(output_dir, "experiment_log.csv")
    log_df.to_csv(log_path, index=False)
    logging.info("Saved experiment log to %s", log_path)

    best_model_name = results_df.sort_values("pr_auc_mean", ascending=False).iloc[0]["model"]
    best_model_path = os.path.join(output_dir, "best_model.joblib")
    dump(fitted_models[best_model_name], best_model_path)
    logging.info("Saved best model (%s) to %s", best_model_name, best_model_path)

    top3 = get_top3_models_by_pr_auc(fitted_models, X_test, y_test)

    pr_path = os.path.join(output_dir, "pr_curves.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, _ in top3:
        PrecisionRecallDisplay.from_estimator(
            fitted_models[name],
            X_test,
            y_test,
            name=name,
            ax=ax,
        )
    ax.set_title("Precision-Recall Curves — Top 3 Models")
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()
    logging.info("Saved PR curves to %s", pr_path)

    calibration_path = os.path.join(output_dir, "calibration.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, _ in top3:
        CalibrationDisplay.from_estimator(
            fitted_models[name],
            X_test,
            y_test,
            n_bins=10,
            name=name,
            ax=ax,
        )
    ax.set_title("Calibration Curves — Top 3 Models")
    plt.tight_layout()
    plt.savefig(calibration_path)
    plt.close()
    logging.info("Saved calibration plot to %s", calibration_path)


def dry_run(df, args):
    logging.info("DRY RUN: no models will be trained.")
    logging.info("Data path: %s", args.data_path)
    logging.info("Output directory: %s", args.output_dir)
    logging.info("Cross-validation folds: %s", args.n_folds)
    logging.info("Random seed: %s", args.random_seed)
    logging.info("Features: %s", NUMERIC_FEATURES)
    logging.info("Target column: %s", TARGET_COLUMN)
    logging.info("Models: Dummy, LR_default, LR_balanced, DT_depth5, RF_default, RF_balanced")


def train_and_evaluate(args):
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.data_path)
    validate_data(df)

    if args.dry_run:
        dry_run(df, args)
        return

    X = df[NUMERIC_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=args.random_seed,
    )

    logging.info(
        "Split complete: %d train, %d test, train churn rate %.2f%%",
        len(X_train),
        len(X_test),
        y_train.mean() * 100,
    )

    models = define_models(args.random_seed)

    results_df = run_cv_comparison(
        models=models,
        X=X_train,
        y=y_train,
        n_folds=args.n_folds,
        random_seed=args.random_seed,
    )

    fitted_models = fit_models(models, X_train, y_train)

    save_outputs(
        results_df=results_df,
        fitted_models=fitted_models,
        X_test=X_test,
        y_test=y_test,
        output_dir=args.output_dir,
    )

    logging.info("Pipeline completed successfully.")


def main():
    setup_logging()
    args = parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()