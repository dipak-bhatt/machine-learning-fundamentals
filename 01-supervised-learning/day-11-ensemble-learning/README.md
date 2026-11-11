# Day 11 — Ensemble Learning

**Module:** Module 2: Supervised Learning

One tree is unstable; one line is rigid. An **ensemble** trains many models and
combines their predictions, and the combination is often better than any member.
The reason is statistical: if the members make *different* mistakes, averaging
cancels some of them. Bagging fights variance, boosting fights bias, and the trick
in both cases is diversity.

This day covers the two dominant families: **random forests**, which bag many
decorrelated trees, and **gradient boosting**, which builds trees in sequence to
correct the current ensemble's errors. These methods are the default choice for
tabular data and the backbone of modern applied machine learning.

## What You Will Learn

- Why combining models can improve accuracy and stability.
- How the bias-variance trade-off explains bagging and boosting.
- How bagging uses bootstrap samples and why it reduces variance.
- How random forests add feature randomness and why that matters.
- How to read the out-of-bag score.
- How gradient boosting fits successive trees to residuals.
- How `n_estimators`, `learning_rate`, and tree depth interact.
- What XGBoost, LightGBM, and CatBoost add to the core idea.
- How voting and stacking combine different model families.
- How to interpret ensembles with impurity and permutation importance.
- When an ensemble is worth its cost, and when a simple baseline wins.

## Detailed Content

- [Theory](theory.md) — why ensembles work, bagging, random forests, OOB error,
  boosting, gradient boosting in detail, modern boosting libraries, voting and
  stacking, strengths and weaknesses, interpretability, use cases, and common
  mistakes.
- [Practical](practical.md) — compare a dummy baseline, logistic regression, and a
  tree, train random forest and gradient boosting models, inspect OOB scores,
  cross-validate, tune the main knobs, compare feature importance methods, and
  combine models with voting and stacking.

## Files

No CSV is needed. The practical uses the Breast Cancer Wisconsin dataset bundled
with scikit-learn:

- **Breast Cancer Wisconsin** (`load_breast_cancer`) — 569 rows, 30 numeric
  features, two classes. It is small enough to train many ensembles quickly and
  clean enough to expose the difference between a single tree and a forest.

The notebook should be created inside this Day 11 folder. Nothing is downloaded.

## Before You Start

- Day 09: decision trees, impurity, overfitting, and pruning.
- Day 10: an alternative classifier and the value of honest comparison.
- Day 04: train/test splits, cross-validation, bias, and variance.

## Outcome

Learners can explain bagging and boosting, train and tune random forests and
gradient boosting, interpret out-of-bag error and cross-validation spread, compare
feature importance methods, combine models with voting and stacking, and justify
when an ensemble is preferable to a simpler baseline.
