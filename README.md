# Integration 5B — Model Comparison & Decision Memo

Module 5 Week B Integration Task for AISPIRE Applied AI & ML Systems.

## Setup

```bash
pip install -r requirements.txt
git checkout -b integration-5b-model-comparison
```

## Tasks

Complete the 9 functions in `model_comparison.py`. The tasks (described in the integration guide) build on each other:

1. `load_and_preprocess` — Load dataset, select features, split 80/20 with stratification
2. `define_models` — Define 6 model configurations as sklearn Pipelines (2×2 default/balanced + Dummy + DT)
3. `run_cv_comparison` — 5-fold stratified CV on all models; compute mean ± std for 5 metrics
4. `save_comparison_table` — Save results DataFrame to CSV
5. `plot_pr_curves_top3` — PR curves for top 3 models on one plot
6. `plot_calibration_top3` — Calibration curves for top 3 models
7. `save_best_model` — Persist best model with `from joblib import dump`
8. `log_experiment` — Log all model results with timestamps to CSV
9. `find_tree_vs_linear_disagreement` — Find one sample where RF and LR disagree most

Run the full script: `python model_comparison.py`
Run tests: `pytest tests/ -v`

## Submission

Your PR description must include:

1. Model comparison table (all 6 models × 5 metrics)
2. Decision memo (3–4 paragraphs recommending a model for Petra Telecom)
3. Paste your PR URL into TalentLMS → Module 5 Week B → Integration Task to submit this assignment.

---

## License

This repository is provided for educational use only. See [LICENSE](LICENSE) for terms.

You may clone and modify this repository for personal learning and practice, and reference code you wrote here in your professional portfolio. Redistribution outside this course is not permitted.
