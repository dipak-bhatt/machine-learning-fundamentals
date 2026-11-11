# Day 17 — PCA and Dimensionality Reduction

**Module:** Module 4: Unsupervised Learning

Real datasets often have more columns than rows, or columns that are expensive to
collect and highly correlated. More features can hurt: they add noise, slow training,
make visualization impossible, and spread points so thin that distance-based methods
fail. **Principal component analysis (PCA)** compresses many features into a few
uncorrelated ones that retain most of the variation.

This day explains why high dimensions are hard, how PCA finds the directions that
matter, how to choose how many components to keep, and how to use PCA without
leaking information. It also covers what PCA cannot do, because PCA is powerful but
not universal.

## What You Will Learn

- What the curse of dimensionality is and why it matters.
- The difference between feature selection and feature extraction.
- How PCA finds orthogonal directions of maximum variance.
- Why standardization is essential before PCA.
- How to read a scree plot and a cumulative explained variance plot.
- How to choose the number of components for a target variance.
- How to measure reconstruction error.
- How to use PCA for visualization and as preprocessing.
- How to interpret component loadings, and why interpretation is limited.
- How to fit PCA inside a pipeline to prevent leakage.
- When PCA helps, when it hurts, and what it cannot represent.

## Detailed Content

- [Theory](theory.md) — the curse of dimensionality, selection versus extraction,
  the PCA algorithm, covariance and eigenvectors, standardization, explained
  variance, reconstruction, visualization, downstream use, leakage, limitations,
  applications, and common mistakes.
- [Practical](practical.md) — scale the breast cancer dataset, fit PCA, plot scree
  and cumulative variance, choose 95% variance, project to two dimensions, compare
  logistic regression and KNN with and without PCA, measure reconstruction error,
  interpret loadings, visualize iris, and cross-validate a leak-free PCA pipeline.

## Files

No CSV is needed. The practical uses two built-in datasets:

- **Breast cancer** (`load_breast_cancer`) — 569 rows, 30 numeric features, binary
  target.
- **Iris** (`load_iris`) — 150 rows, 4 numeric features, 3 classes, for
  visualization.

Both load through scikit-learn, so no download is required.

## Before You Start

- Day 15: clustering and evaluation.
- Day 14: distance metrics and high-dimensional behavior.
- Day 04: leakage and train/test separation.
- Day 05 or Day 08: pipelines and logistic regression.

## Outcome

Learners can explain dimensionality reduction, fit and interpret PCA, choose a
component count from explained variance, measure reconstruction error, use PCA for
visualization and preprocessing, and apply it inside a leak-free pipeline while
recognizing its limitations.
