# Day 09 — Decision Trees

**Module:** Module 2: Supervised Learning

Every model so far has produced a formula: a line, a curve, or a stack of weighted
sums. A **decision tree** instead produces a flowchart. It asks a sequence of
yes/no questions about the features and follows the answers to a leaf, where the
prediction waits. That change of representation buys automatic interactions,
freedom from scaling, and an explanation a person can read out loud. It also brings
new risks: a tree that is left unchecked will memorize the training set, and its
questions can change completely when a few rows change.

Today we build trees from the ground up: the parts of a tree, entropy and Gini
impurity, information gain, the greedy learning algorithm, and the pruning that
keeps a tree honest.

## What You Will Learn

- What a decision tree is and how it differs from a linear model.
- The parts of a tree: root, internal nodes, branches, leaves, and splitting rules.
- Which problems suit decision trees, and which do not.
- How a tree routes a row to a leaf, and how to read that decision path.
- How entropy and Gini impurity measure the mixedness of a node.
- How information gain scores a candidate split.
- How the greedy recursive partitioning algorithm grows a tree.
- Why an unconstrained tree overfits, and how depth, leaf size, and pruning help.
- How trees handle regression, and why they cannot extrapolate.
- How to read impurity-based and permutation feature importance.
- When a single tree is the right tool and when an ensemble is better.

## Detailed Content

- [Theory](theory.md) — tree representation, appropriate problems, entropy, Gini,
  information gain, a hand-worked split, the greedy learning algorithm, stopping
  rules, pruning and cost complexity, regression trees, feature importance,
  instability, strengths and limitations, and common mistakes.
- [Practical](practical.md) — compute impurity and gain by hand, train and visualize
  an Iris tree, read a decision path, sweep `max_depth`, prune with `ccp_alpha`,
  rank features two ways, and compare a regression tree with a linear model on the
  Day 04 house data.

## Files

- **Iris** (`sklearn.datasets.load_iris`) — 150 flowers, four measurements, three
  species. Bundled with scikit-learn, so nothing is downloaded.
- `../../00-foundations/day-04-bias-variance-ml-workflow/house_prices.csv` — the
  140-house synthetic dataset used earlier in the course, with `area_sqft`,
  `age_years` (6 missing), `location`, and `price_usd`, used here for the
  regression tree comparison.

The notebook should be created inside this Day 09 folder so the relative path to
the CSV works.

## Before You Start

- Day 07: classification labels, decision boundaries, and classification metrics.
- Day 04: train/test splits, baselines, overfitting, and honest evaluation.
- Day 05: a linear baseline, residuals, and RMSE, used in the regression section.

## Outcome

Learners can describe a tree in terms of nodes, branches, and leaves, compute
entropy, Gini impurity, and information gain for a split, explain the greedy tree
learning algorithm, control overfitting with depth, leaf-size, and cost-complexity
pruning, read a decision path and feature importances, and compare a tree with a
linear model on held-out data.
