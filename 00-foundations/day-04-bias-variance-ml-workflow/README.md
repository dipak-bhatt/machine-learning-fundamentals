# Day 04 — Bias-Variance Tradeoff and the ML Workflow

**Module:** Module 1: Foundations

A model can be too simple and consistently wrong, or too complex and unstable.
Today we learn to recognize both problems, balance them, and follow a safe
end-to-end workflow so that our results are honest and reproducible.

See the detailed content in:

- [Theory](theory.md) — bias, variance, irreducible noise, the bias-variance
  tradeoff, model complexity, regularization, random seeds, data leakage, the
  full ML workflow, and real-world use cases.
- [Practical](practical.md) — train simple and complex trees on
  `house_prices.csv`, visualize underfitting and overfitting, compare training
  and test RMSE, and build a preprocessing and training pipeline.

## Files

- `house_prices.csv` — a synthetic dataset of 140 houses with area, age,
  location, and price. It includes a few missing ages so we can practice safe
  imputation inside a pipeline.

## Outcome

Learners can explain the bias-variance tradeoff with a concrete example,
recognize underfitting and overfitting, choose model complexity using unseen
data, avoid data leakage, and describe the complete machine learning lifecycle.
