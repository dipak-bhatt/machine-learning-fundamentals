# Day 18 — Classification Metrics

**Module:** Module 5: Evaluation and Optimization

Accuracy is the first metric everyone learns and the one most likely to mislead.
When one class dominates, a model that ignores the minority can look excellent while
being useless. This day replaces the single number with a small family of metrics
built from the confusion matrix: precision, recall, specificity, F1, and their
multi-class averages.

The goal is not to memorize formulas. It is to connect each metric to the cost of a
particular kind of mistake, so that the evaluation matches the decision the model is
meant to support.

## What You Will Learn

- How to build and read a binary and multi-class confusion matrix.
- What true positives, false positives, false negatives, and true negatives are.
- How precision, recall, specificity, F1, and balanced accuracy are computed.
- Why accuracy fails under class imbalance: the accuracy paradox.
- How `pos_label` changes which class a metric describes.
- How to compare macro, weighted, and micro averages.
- How class weights and thresholds trade precision for recall.
- How to choose a threshold from a cost matrix.
- How to report classification results without hiding the errors.

## Detailed Content

- [Theory](theory.md) — the confusion matrix, accuracy, precision, recall,
  specificity, F1 and F-beta, multi-class averaging, metric selection by error cost,
  class imbalance, the precision-recall trade-off, a brief note on ROC/AUC, reporting
  practices, and common mistakes.
- [Practical](practical.md) — compute metrics on breast cancer, demonstrate the
  accuracy paradox with a dummy classifier, compare default and class-weighted
  logistic regression, sweep thresholds, minimize a cost function, and evaluate iris
  with macro, weighted, and micro averages.

## Files

No CSV is needed. The practical uses:

- **Breast cancer** (`load_breast_cancer`) — a mildly imbalanced binary problem.
- **A synthetic imbalanced dataset** (`make_classification`) — 95/5 class weights.
- **Iris** (`load_iris`) — a three-class problem for averaging.

All load through scikit-learn, so no download is required.

## Before You Start

- Day 17: PCA and leak-free pipelines.
- Day 05 or Day 08: logistic regression and train/test evaluation.
- Day 04: the train/test split and why it matters.

## Outcome

Learners can build a confusion matrix, compute and interpret precision, recall,
specificity, F1, and balanced accuracy, diagnose class imbalance, apply class weights
and threshold tuning, choose a threshold from error costs, and report multi-class
results with an explicit averaging choice.
