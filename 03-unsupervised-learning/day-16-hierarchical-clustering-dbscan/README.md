# Day 16 — Hierarchical Clustering and Modern Clustering

**Module:** Module 4: Unsupervised Learning

K-means assumes compact, spherical clusters of similar size. Real data often
disagree: two moons interlock, rings wrap around each other, and a few points sit far
from everything. This day adds two methods that make different assumptions.
**Hierarchical clustering** builds a tree of merges, so the number of clusters can be
chosen after the fact by cutting the tree. **DBSCAN** defines clusters as dense
regions, so it can follow arbitrary shapes and label sparse points as noise.

The lesson is not that these methods are better. It is that every clustering
algorithm encodes a definition of "cluster", and the data decide which definition
fits.

## What You Will Learn

- Why K-means fails on non-spherical shapes.
- How agglomerative and divisive hierarchical clustering work.
- What single, complete, average, and Ward linkage optimize.
- How to read a dendrogram and cut it at a chosen height or cluster count.
- How distance metrics and scaling affect hierarchical clustering.
- What core, border, and noise points are in DBSCAN.
- How `eps` and `min_samples` control the density definition.
- How to compare centroid, hierarchical, and density-based clustering.
- How to use cluster labels as features without leaking test information.
- Which clustering family fits which kind of data.

## Detailed Content

- [Theory](theory.md) — limitations of K-means, hierarchical clustering, linkage
  methods, dendrograms, distance metrics, DBSCAN, density versus centroid methods,
  clustering before supervised learning, applications, strengths, limitations, and
  common mistakes.
- [Practical](practical.md) — show K-means failing on moons and circles, compare four
  linkage methods, draw and cut a dendrogram, tune DBSCAN on both shapes, compare
  all methods in one table, and demonstrate the correct clustering-before-
  classification workflow.

## Files

No CSV is needed. The practical generates:

- **Two moons** (`make_moons`) — two interleaving crescents.
- **Concentric circles** (`make_circles`) — an inner ring surrounded by an outer
  ring.

True labels are used only to compute the adjusted Rand index for comparison, never
as input to the clustering.

The notebook should be created inside this Day 16 folder. Nothing is downloaded.

## Before You Start

- Day 15: clustering, K-means, scaling, and cluster evaluation.
- Day 14: distance metrics and the curse of dimensionality.
- Day 04: leakage, train/test separation, and honest evaluation.

## Outcome

Learners can apply agglomerative clustering with different linkages, read and cut a
dendrogram, run DBSCAN and interpret clusters versus noise, tune its two parameters,
choose among K-means, hierarchical, and density-based methods, and add cluster
features to a supervised model without contaminating evaluation.
