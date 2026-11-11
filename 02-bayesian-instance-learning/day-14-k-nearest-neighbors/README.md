# Day 14 — K-Nearest Neighbors (KNN)

**Module:** Module 3: Bayesian and Instance-Based Learning

Some models learn a compact formula. **K-nearest neighbors** does not. It keeps the
training data and, for each new point, finds the most similar stored examples and
lets them vote. There is no global equation, no parameter vector, and almost no
training phase. The model is the data.

That simplicity forces a set of choices: how to measure distance, how many
neighbors to consult, whether close neighbors should count more, and how to handle
the fact that distances lose meaning in high dimensions. KNN is a good vehicle for
learning those choices because every one of them is visible.

## What You Will Learn

- What instance-based and lazy learning mean.
- How a KNN prediction is assembled from neighbors.
- How to compute Euclidean, Manhattan, and Minkowski distances.
- How `k` controls the bias-variance trade-off.
- Why cross-validation is necessary to choose `k`.
- How distance-weighted voting changes predictions.
- Why feature scaling is mandatory.
- What the curse of dimensionality is and how it breaks KNN.
- How KNN handles multiclass classification and regression.
- When KNN is a strong choice and when it is the wrong tool.

## Detailed Content

- [Theory](theory.md) — instance-based learning, the prediction rule, distance
  metrics, choosing `k`, distance weighting, scaling, the curse of dimensionality,
  KNN regression, strengths, limitations, use cases, and common mistakes.
- [Practical](practical.md) — compute distances by hand, sweep `k` on Iris, choose
  `k` with cross-validation, plot decision boundaries, compare metrics on two
  moons, test distance weighting, show a scaling failure on breast cancer, and
  demonstrate the curse of dimensionality.

## Files

No CSV is needed. The practical uses:

- **Iris** (`load_iris`) — 150 flowers, four features, three balanced classes.
- **Two moons** (`make_moons`) — a curved binary problem for comparing metrics.
- **Breast Cancer Wisconsin** (`load_breast_cancer`) — 569 rows and 30 features for
  the scaling experiment.
- **Random points** — generated to show how distances behave in high dimensions.

The notebook should be created inside this Day 14 folder. Nothing is downloaded.


## Before You Start

- Day 07: classification and decision boundaries.
- Day 09: axis-aligned boundaries and the idea of local prediction.
- Day 04: train/test splits, cross-validation, bias, and variance.

## Outcome

Learners can explain how KNN predicts from nearby examples, compute common distance
metrics, choose `k` with cross-validation, apply distance weighting and scaling,
describe the curse of dimensionality, and decide when KNN is a sensible model.
