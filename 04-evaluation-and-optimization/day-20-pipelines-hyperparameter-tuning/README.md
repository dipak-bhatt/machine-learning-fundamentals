# Day 20 — Preprocessing, Hyperparameter Tuning, and Responsible ML

**Module:** Module 5: Evaluation and Optimization

A model is more than its algorithm. It includes the imputation, encoding, scaling, and
feature selection that prepare the data. When those steps are done by hand, they are
easy to get wrong and almost impossible to cross-validate honestly. A `Pipeline`
makes the whole sequence a single object that fits, transforms, and predicts as one.

This day builds that object, then tunes it. It covers the difference between
parameters and hyperparameters, grid and randomized search, and the selection bias
that appears when tuning and evaluation share data. It closes with the responsible-use
questions that no metric can answer: fairness across groups, privacy, explainability,
and appropriate scope.

## What You Will Learn

- How `Pipeline` and `ColumnTransformer` organize preprocessing.
- How to impute, encode, and scale numeric and categorical columns.
- Why `handle_unknown="ignore"` matters in production.
- The difference between learned parameters and chosen hyperparameters.
- How `GridSearchCV` and `RandomizedSearchCV` work.
- Why tuning does not always improve a model.
- How selection bias arises and how nested CV addresses it.
- How to compare tuned and untuned models honestly.
- How to inspect feature importance and audit subgroup performance.
- How target encoding outside the folds leaks information.
- The responsible-ML dimensions of fairness, privacy, explainability, and scope.

## Detailed Content

- [Theory](theory.md) — manual preprocessing failures, the pipeline abstraction,
  `ColumnTransformer`, imputation, encoding, scaling, feature engineering, parameters
  versus hyperparameters, grid and randomized search, selection bias and nested CV,
  resampling inside pipelines, fairness, privacy, explainability, accountability, and
  common mistakes.
- [Practical](practical.md) — build a mixed-type dataset with missing values, construct
  a `ColumnTransformer`, cross-validate a logistic pipeline, grid-search logistic
  regression and a decision tree, run randomized search, read feature importances,
  evaluate by subgroup, and demonstrate target-encoding leakage.

## Files

No CSV is needed. The practical generates a synthetic churn dataset:

- 1,200 rows, six numeric features, two categorical features.
- 60 missing values in each of two numeric columns and in one categorical column.
- A binary target with roughly balanced classes.

A second small random dataset with a high-cardinality category is used for the leakage
demonstration. Everything is generated locally.

## Before You Start

- Day 19: cross-validation and leakage.
- Day 18: classification metrics.
- Day 17: pipelines and preprocessing.
- Day 11: decision trees and why depth matters.

## Outcome

Learners can build reproducible preprocessing-and-model pipelines, tune them with grid
and randomized search inside cross-validation, explain when tuning helps and when it
does not, detect and prevent preprocessing leakage, inspect feature importance, audit
performance across subgroups, and reason about fairness, privacy, explainability, and
appropriate use.
