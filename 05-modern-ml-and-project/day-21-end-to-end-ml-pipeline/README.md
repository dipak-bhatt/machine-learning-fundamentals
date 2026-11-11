# Day 21 — End-to-End ML Pipeline

**Module:** Module 6: Modern ML and Project

Individual techniques are not a project. A real project moves through problem
definition, data inspection, splitting, preprocessing, baseline modeling, model
selection, final evaluation, serialization, deployment, and monitoring. Each stage
depends on the others, and shortcuts at one stage surface as failures at another.

This day assembles the pieces from the previous weeks into a single reproducible
pipeline. It emphasizes the habits that separate a working model from a notebook
experiment: a baseline, a leak-free preprocessing chain, a test set touched once,
fixed seeds, and a saved artifact with metadata.

## What You Will Learn

- The stages of the ML lifecycle and what each one decides.
- How to write a problem statement with target, unit, metric, and constraints.
- Why the data is split before any evaluation.
- How to package preprocessing and model into one pipeline.
- Why a trivial baseline is essential.
- How to compare candidate models with cross-validation.
- Why the test set is evaluated once.
- How to make an experiment reproducible with seeds and recorded configuration.
- How to save and load a complete pipeline with `joblib`.
- How to predict on new records that contain missing values or unseen categories.
- Why model versioning, experiment tracking, and monitoring matter.

## Detailed Content

- [Theory](theory.md) — the lifecycle, problem definition, point-in-time correctness,
  data splitting, preprocessing, baselines, model selection, final evaluation,
  reproducibility, serialization, versioning, experiment tracking, monitoring,
  retraining, and common mistakes.
- [Practical](practical.md) — load and inspect the house-price CSV, define the
  problem, split the data, build a `ColumnTransformer` pipeline, evaluate a mean
  baseline and linear regression, compare random forest and gradient boosting with
  cross-validation, evaluate the selected model once on test, save and reload it with
  `joblib`, and predict new houses with missing values.

## Files

- `../../00-foundations/day-04-bias-variance-ml-workflow/house_prices.csv` — 140 rows,
  4 columns (`area_sqft`, `age_years`, `location`, `price_usd`), 6 missing ages.
- `house_price_pipeline.joblib` — created by the practical when you run it.
- Metadata printed by the practical, to be saved alongside the model in a real
  project.

No download is required.

## Before You Start

- Day 20: pipelines, `ColumnTransformer`, and hyperparameter tuning.
- Day 19: cross-validation and leakage.
- Day 04: the train/test split and bias-variance trade-off.
- Day 11: random forests and gradient boosting.

## Outcome

Learners can turn a dataset into a reproducible end-to-end pipeline: define the
problem and metric, split before evaluating, build leak-free preprocessing, establish
baselines, select a model on validation data, evaluate once on test, serialize the
full pipeline, predict on new records, and attach the metadata needed to reproduce and
maintain the result.
