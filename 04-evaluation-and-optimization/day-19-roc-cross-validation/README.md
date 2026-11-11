# Day 19 — ROC Curves and Cross-Validation

**Module:** Module 5: Evaluation and Optimization

A single accuracy number hides two things: how the model behaves at other thresholds,
and how much the score depends on the particular train/test split. This day fixes
both gaps. **ROC curves** and **precision-recall curves** show performance across
every threshold. **Cross-validation** averages performance over several splits so the
estimate is stable.

The day also shows the failure modes. ROC can look flattering when positives are rare,
and cross-validation can be quietly optimistic when preprocessing is done outside the
folds. Knowing both is what makes an evaluation trustworthy.

## What You Will Learn

- How to build an ROC curve from scores and thresholds.
- What true positive rate, false positive rate, and AUC mean.
- Why a random classifier has AUC 0.5.
- How to build a precision-recall curve and compute average precision.
- When PR is more informative than ROC under class imbalance.
- How k-fold and stratified k-fold cross-validation work.
- How to read the mean and spread of CV scores.
- Why stratified folds reduce variance on imbalanced data.
- How repeated CV gives a more stable estimate.
- How preprocessing outside CV causes leakage and inflates scores.
- Why nested CV is needed when hyperparameters are tuned.

## Detailed Content

- [Theory](theory.md) — scores and thresholds, ROC construction, AUC interpretation,
  ROC under imbalance, precision-recall curves and average precision, k-fold,
  stratified, leave-one-out, repeated, group and time-series CV, reading CV results,
  leakage in CV, nested CV, and common mistakes.
- [Practical](practical.md) — plot ROC and compute AUC on breast cancer, compare
  logistic regression, KNN, and a dummy, plot PR curves and average precision,
  contrast ROC and PR on imbalanced data, run k-fold and stratified CV, collect
  multiple metrics, compare `KFold` and `StratifiedKFold`, repeat CV, and demonstrate
  feature-selection leakage.

## Files

No CSV is needed. The practical uses:

- **Breast cancer** (`load_breast_cancer`) — a mildly imbalanced binary problem.
- **Synthetic imbalanced data** (`make_classification`) — 95/5 and 90/10 class
  weights.
- **Synthetic high-dimensional noise** — 200 samples and 2,000 features for the
  leakage demonstration.

All load through scikit-learn, so no download is required.

## Before You Start

- Day 18: confusion matrix, precision, recall, F1, and thresholds.
- Day 17: pipelines and leak-free preprocessing.
- Day 04: why the train/test split exists and what leakage is.

## Outcome

Learners can construct and interpret ROC and precision-recall curves, explain AUC and
average precision, select the right curve for imbalanced data, run and read k-fold,
stratified, and repeated cross-validation, and prevent preprocessing leakage by using
pipelines inside the CV loop.
