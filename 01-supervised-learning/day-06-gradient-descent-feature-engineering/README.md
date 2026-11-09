# Day 06 — Gradient Descent and Feature Engineering

**Module:** Module 2: Supervised Learning

On Day 05, scikit-learn found the best regression line for us. Today we open the
box: **gradient descent** is the repeated calculation that improves a model's
parameters step by step. We then change the inputs themselves, using **feature
scaling**, **polynomial features**, and **regularization** (Ridge and Lasso), so
that a linear model can handle more realistic patterns.

## What You Will Learn

- How a loss function turns prediction mistakes into one number to minimize.
- What a gradient is, and why we subtract it from the parameters.
- How to derive and implement the gradient descent update for linear regression.
- How the learning rate affects convergence, and how to read a loss curve.
- Why feature scaling helps gradient descent and regularized models.
- How polynomial features let a line-based model fit curves.
- How Ridge shrinks coefficients and how Lasso can remove features.
- How to compare every model on the same held-out rows against a baseline.

## Detailed Content

- [Theory](theory.md) — closed-form versus iterative solutions, MSE, the loss
  surface, derivatives and the gradient, full derivations, batch versus stochastic
  descent, learning rate, scaling, polynomial features, multicollinearity, Ridge,
  Lasso, Elastic Net, evaluation, and common mistakes.
- [Practical](practical.md) — perform an update by hand, implement batch gradient
  descent in NumPy, tune the learning rate, and compare plain, scaled,
  polynomial, Ridge, and Lasso regression on the Day 04 house data.

## Files

- `../../00-foundations/day-04-bias-variance-ml-workflow/house_prices.csv` — the
  140-house synthetic dataset used earlier in the course, with `area_sqft`,
  `age_years` (6 missing), `location`, and `price_usd`. Reusing it lets us focus
  on the new methods instead of learning another dataset.

The notebook for the practical should be created inside this Day 06 folder so the
relative path to the CSV works.

## Before You Start

- Day 05: a line, residuals, least squares, and MAE/RMSE/R².
- Day 04: train/test splits, data leakage, and the ML workflow.

## Outcome

Learners can explain how gradient descent reduces prediction error, implement a
batch update rule, interpret a loss curve, explain why scaling helps, create
polynomial features, and compare ordinary, Ridge, and Lasso regression using
honest held-out evaluation.
