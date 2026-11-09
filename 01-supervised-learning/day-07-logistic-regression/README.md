# Day 07 — Logistic Regression and Classification

**Module:** Module 2: Supervised Learning

For the last two days we have predicted numbers. But many of the most useful
questions have a **choice** as the answer: Is this tumor malignant or benign?
Is this email spam? Will this customer cancel? **Logistic regression** adapts the
linear model to these questions. It computes a linear score, converts it into a
probability with the **sigmoid**, and then turns that probability into a class
label using a threshold.

Today we learn why classification needs its own loss and metrics, and how to read a
model's mistakes honestly.

## What You Will Learn

- The difference between regression and classification, and between binary and
  multiclass problems.
- How the sigmoid turns a linear score into a probability between 0 and 1.
- How to interpret coefficients as changes in log-odds and as odds ratios.
- What log loss measures, and why it punishes confident mistakes.
- How the decision boundary and classification threshold control the tradeoff
  between false positives and false negatives.
- How to read a confusion matrix and compute precision, recall, specificity, and F1.
- Why accuracy is misleading when classes are imbalanced.
- How one-vs-rest and multinomial strategies extend logistic regression to more
  than two classes.

## Detailed Content

- [Theory](theory.md) — regression versus classification, why a line is the wrong
  model, the sigmoid, odds and log-odds, decision boundaries, log loss, the gradient
  derivation, regularization, coefficient interpretation, evaluation metrics, class
  imbalance, multiclass strategies, assumptions, and common mistakes.
- [Practical](practical.md) — compute the sigmoid and one update by hand, train
  logistic regression on the Breast Cancer Wisconsin data, inspect probabilities,
  move the threshold, read the confusion matrix, convert coefficients to odds
  ratios, and train a multiclass model on Iris.

## Files

No CSV is needed. The practical uses two datasets that ship with scikit-learn:

- **Breast Cancer Wisconsin** — 569 samples, 30 numeric features, and a binary
  target (malignant or benign). Used for the main classification example.
- **Iris** — 150 flowers, four features, and three species. Used for the multiclass
  example.

Both are loaded automatically, so the notebook can live in this Day 07 folder
without any path configuration or download step.

## Before You Start

- Day 05: lines, residuals, and the difference between a prediction and a target.
- Day 06: gradients, feature scaling, and regularization.
- Day 04: train/test splits, data leakage, baselines, and honest evaluation.

## Outcome

Learners can explain how logistic regression produces a probability, interpret
coefficients as odds ratios, explain log loss and the gradient, choose a
classification threshold based on the cost of mistakes, evaluate a classifier with
a confusion matrix, precision, recall, and F1, handle class imbalance, and train
both binary and multiclass logistic regression using honest held-out evaluation.
