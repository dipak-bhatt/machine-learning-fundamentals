# Day 15 — Clustering and K-Means

**Module:** Module 4: Unsupervised Learning

Until now every row had an answer to learn from. **Unsupervised learning** removes
that answer and asks a different question: what structure is already here? The most
common answer is **clustering**, which groups rows so that members of a group are
more similar to each other than to the rest.

**K-means** is the simplest and most used clustering algorithm. It alternates two
steps, assign points to the nearest centroid and move each centroid to the mean of
its points, until the assignments stop changing. The result is fast, interpretable,
and easy to visualize, but it comes with assumptions that are easy to violate. This
day covers both the method and its failure modes.

## What You Will Learn

- The difference between supervised and unsupervised learning.
- What clustering is used for and why evaluation is harder without labels.
- The K-means assignment and update steps.
- The within-cluster sum of squares (inertia) objective.
- Why K-means converges to a local minimum.
- How to choose `k` with the elbow method and silhouette score.
- How initialization and k-means++ affect the result.
- Why multiple restarts (`n_init`) matter.
- Why scaling and outliers change the clusters.
- When K-means is the wrong algorithm.

## Detailed Content

- [Theory](theory.md) — supervised versus unsupervised learning, clustering
  applications, the K-means algorithm, WCSS, local minima, choosing `k`, k-means++,
  assumptions, scaling, outliers, evaluation without labels, a customer
  segmentation case, and common mistakes.
- [Practical](practical.md) — one assignment-update iteration by hand, cluster four
  blobs, apply the elbow and silhouette methods, compare initializations, cluster
  Iris without labels, show scaling change customer segments, and measure outlier
  sensitivity.

## Files

No CSV is needed. The practical uses:

- **Four blobs** (`make_blobs`) — 300 generated points in four clear groups.
- **Iris** (`load_iris`) — 150 flowers with labels used only for checking, not for
  clustering.
- **Generated customer data** — income and spending values with very different
  scales, for the standardization experiment.

The notebook should be created inside this Day 15 folder. Nothing is downloaded.

## Before You Start

- Day 09: impurity, local greedy decisions, and overfitting.
- Day 14: distance metrics and feature scaling.
- Day 04: why evaluation must use held-out or honest measures.

## Outcome

Learners can run K-means, interpret centroids and inertia, choose `k` with the
elbow method and silhouette score, explain initialization and the need for multiple
restarts, identify scaling and outlier problems, evaluate clusters without labels,
and decide when a density-based method is more appropriate.
