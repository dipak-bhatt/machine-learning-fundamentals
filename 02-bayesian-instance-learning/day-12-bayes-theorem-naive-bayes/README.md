# Day 12 — Probability, Bayes' Rule, and Naive Bayes

**Module:** Module 3: Bayesian and Instance-Based Learning

Bayesian reasoning treats belief as a probability and updates it when evidence
arrives. The update rule is **Bayes' theorem**, and it produces one of the most
useful surprises in statistics: a 95%-accurate test for a rare disease can still
leave an 8.8% chance that a positive result is real, because false positives from
the healthy majority swamp the true positives.

This day builds the probability language needed to read that formula, applies it to
diagnosis, and then turns it into a classifier. **Naive Bayes** makes a bold
independence assumption that is almost always false, yet it remains a fast and
strong baseline for spam filtering, sentiment analysis, and other text problems.

## What You Will Learn

- The rules of probability, events, and complements.
- Conditional, joint, marginal, and independent probabilities.
- How to apply Bayes' theorem and name the prior, likelihood, evidence, and
  posterior.
- Why base rates dominate diagnostic reasoning.
- How MAP classification chooses a class from posterior probabilities.
- How Naive Bayes factorizes the likelihood with a conditional-independence
  assumption.
- Why the naive assumption still produces useful classifiers.
- The difference between Multinomial, Bernoulli, and Gaussian Naive Bayes.
- How Laplace smoothing solves the zero-frequency problem.
- How to read the words that drive a text classifier.
- When Naive Bayes is a sensible choice and when it is not.

## Detailed Content

- [Theory](theory.md) — probability rules, conditional probability, joint and
  marginal distributions, independence, Bayes' theorem, Bayesian decision theory,
  the Naive Bayes model, its three variants, Laplace smoothing, why it works for
  text, strengths, limitations, and common mistakes.
- [Practical](practical.md) — solve a medical screening problem by hand and by
  simulation, build a bag-of-words matrix, train and interpret a Multinomial Naive
  Bayes spam classifier, study smoothing, and compare Gaussian Naive Bayes with
  logistic regression on breast cancer data.

## Files

No CSV is needed. The practical uses:

- **A small inline corpus** of 16 labelled spam and ham messages, defined in the
  notebook for a transparent bag-of-words example.
- **Breast Cancer Wisconsin** (`load_breast_cancer`) — 569 rows and 30 continuous
  features for the Gaussian variant.

The notebook should be created inside this Day 12 folder. Nothing is downloaded.

## Before You Start

- Day 07: classification, probabilities, and decision boundaries.
- Day 11: comparing models on identical splits.
- Day 04: train/test splits and honest evaluation.

## Outcome

Learners can compute conditional probabilities and apply Bayes' theorem, explain
prior, likelihood, evidence, and posterior, build and interpret a Naive Bayes
classifier, apply smoothing, read word-level evidence, and evaluate a Bayesian
classifier against a stronger baseline with appropriate metrics.
