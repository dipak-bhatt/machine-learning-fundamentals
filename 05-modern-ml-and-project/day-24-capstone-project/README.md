# Day 24 — Capstone Demonstration, Review, and Assessment

**Module:** Module 6: Modern ML and Project

The capstone asks you to do the whole job: define a problem, handle data without
leakage, build a reproducible pipeline, establish a baseline, select a model honestly,
evaluate it once on held-out data, analyze errors, and explain the result to people who
did not write the code.

This day is the assessment. It provides the project brief, a verified reference
skeleton, reproducibility and code-review checklists, a model-card template, a
presentation outline, a demo script, peer-review and viva question banks, and a
rubric. The goal is a project you can defend, not a perfect score.

## What You Will Learn

- How to frame a capstone problem and choose a metric.
- How to structure a complete, reproducible ML project.
- How to build a verified baseline and candidate comparison.
- How to analyze errors and subgroup performance.
- How to document a model with a model card.
- How to present technical work to mixed audiences.
- How to demonstrate predictions on unseen records.
- How to answer conceptual and practical questions about your choices.
- How to assess your own work and a peer's work.
- How to state limitations and responsible-use considerations.

## Detailed Content

- [Theory](theory.md) — what the capstone demonstrates, the complete workflow,
  algorithm selection, evaluation checklist, common mistakes, limitations and
  uncertainty, communication for technical and nontechnical audiences, responsible
  use, what a strong submission looks like, and how to defend the work.
- [Practical](practical.md) — project options, required deliverables, a verified
  breast-cancer reference skeleton, reproducibility and code-review checklists, an
  error-analysis template, a model-card template, a 10–12 minute presentation outline,
  a live demo script, a peer-review form, a viva question bank, and an assessment
  rubric.

## Files

- `capstone_model.joblib` — created by your project when you save the final pipeline.
- Your own data file, or one of the built-in datasets:
  - `load_breast_cancer`, `load_digits`, `load_iris` (scikit-learn).
  - `../../00-foundations/day-04-bias-variance-ml-workflow/house_prices.csv`.

No download is required for the suggested options.

## Before You Start

This capstone draws on every earlier day. In particular:

- Day 04: bias-variance, train/test splits, and the workflow.
- Day 18 and Day 19: classification metrics and cross-validation.
- Day 20: pipelines, `ColumnTransformer`, and tuning.
- Day 21: end-to-end pipeline and `joblib`.
- Day 22 and Day 23: modern models and deployment considerations.

## Outcome

Each learner or group can explain, demonstrate, and defend a complete ML project:
justify the problem, data, model, and metric; show a leak-free evaluation; analyze
errors and subgroups; document limitations and responsible-use considerations; and
communicate the result clearly to both technical and nontechnical audiences.
