# Day 03 — Training, Validation, Testing, and Generalization

**Module:** Module 1: Foundations

A model that scores 100% on the data it has already seen may still be useless on
new data. Today we learn why we must test on unseen data, how to split a dataset
into training and test sets, how to recognize overfitting and underfitting, and
how to read a learning curve — all with simple examples.

See the detailed content in:

- [Theory](theory.md) — training/validation/test sets, training vs test error,
  overfitting, underfitting, generalization, sources of error, learning curves,
  data leakage, and ML use cases for each concept.
- [Practical](practical.md) — split `students_large.csv`, compare training and
  test accuracy, make a model memorize student IDs, and plot learning curves.

## Files

- `students_large.csv` — a synthetic dataset of 200 students with three useful
  features and five irrelevant features, used in the practical.

## Outcome

Learners understand why high training accuracy does not necessarily mean a good
model, and can split data and read basic learning curves.
