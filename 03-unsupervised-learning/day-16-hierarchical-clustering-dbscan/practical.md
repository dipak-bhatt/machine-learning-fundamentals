# Day 16 — Practical: Hierarchical Clustering and DBSCAN

**Module:** Module 4: Unsupervised Learning

In this practical we compare three clustering families on shapes that K-means cannot
handle. We will:

- Show K-means failing on two moons and concentric circles.
- Apply agglomerative clustering with four linkage methods.
- Draw and read a dendrogram.
- Cut the dendrogram with SciPy's `fcluster`.
- Run DBSCAN and interpret clusters versus noise.
- Study the effect of `eps` and `min_samples`.
- Discuss how to use clusters as features without leaking information.

All datasets are generated with scikit-learn, so no download is needed.

## 1. Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

RANDOM_STATE = 42
```

## 2. Generate Non-Spherical Data

Two moons and two concentric circles are the standard stress tests for clustering.

```python
X_moons, y_moons = make_moons(n_samples=400, noise=0.05, random_state=RANDOM_STATE)
X_circles, y_circles = make_circles(
    n_samples=400, noise=0.05, factor=0.5, random_state=RANDOM_STATE
)

print("moons:", X_moons.shape, np.bincount(y_moons))
print("circles:", X_circles.shape, np.bincount(y_circles))
```

Expected output:

```text
moons: (400, 2) [200 200]
circles: (400, 2) [200 200]
```

The labels exist only because we generated the data. We use them to compute the
adjusted Rand index (ARI), where 1 means perfect agreement with the true groups.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap="coolwarm", s=15)
axes[0].set_title("two moons")

axes[1].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap="coolwarm", s=15)
axes[1].set_title("concentric circles")

plt.tight_layout()
plt.show()
```

## 3. K-Means Baseline

K-means minimizes squared distance to centroids, so it can only make a straight cut
between the groups.

```python
for name, X, y in [
    ("moons", X_moons, y_moons),
    ("circles", X_circles, y_circles),
]:
    model = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE)
    labels = model.fit_predict(X)
    print(
        f"{name}: ARI={adjusted_rand_score(y, labels):.4f} "
        f"silhouette={silhouette_score(X, labels):.4f}"
    )
```

Expected output:

```text
moons: ARI=0.2481 silhouette=0.4887
circles: ARI=-0.0021 silhouette=0.3545
```

K-means barely beats chance on the circles and is poor on the moons. The silhouette
scores look moderate, not catastrophic, yet the partitions are wrong: the two halves
are compact but they cut across the true shapes. Geometry alone does not tell you
whether the clusters are meaningful.

## 4. Agglomerative Clustering with Different Linkages

Run agglomerative clustering with `n_clusters=2` and compare linkages.

```python
for linkage_method in ["single", "complete", "average", "ward"]:
    model = AgglomerativeClustering(
        n_clusters=2, linkage=linkage_method
    )
    labels = model.fit_predict(X_moons)
    print(
        f"{linkage_method}: ARI={adjusted_rand_score(y_moons, labels):.4f} "
        f"silhouette={silhouette_score(X_moons, labels):.4f}"
    )
```

Expected output:

```text
single: ARI=1.0000 silhouette=0.3275
complete: ARI=0.4818 silhouette=0.4434
average: ARI=0.4818 silhouette=0.4434
ward: ARI=0.4818 silhouette=0.4434
```

The result is striking. **Single linkage** recovers the moons perfectly, because it
merges clusters through their closest points and can follow the crescent shape.
Complete, average, and Ward measure the far edges of the clusters, so they cut each
moon in half.

Silhouette tells the opposite story: single linkage has the lowest score. The
crescents are long and thin, so points at the ends are far from the cluster centre.
Silhouette rewards compactness, not correct shape. This is a clear case where an
internal metric must be interpreted with the data in mind.

Visualize the two outcomes:

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, method in zip(axes, ["single", "ward"]):
    labels = AgglomerativeClustering(
        n_clusters=2, linkage=method
    ).fit_predict(X_moons)
    ax.scatter(X_moons[:, 0], X_moons[:, 1], c=labels, cmap="coolwarm", s=15)
    ax.set_title(f"agglomerative, {method} linkage")

plt.tight_layout()
plt.show()
```

## 5. Draw a Dendrogram

A dendrogram shows every merge and the distance at which it happened. Use a subset so
the labels remain readable.

```python
subset = X_moons[:60]
Z = linkage(subset, method="single")

plt.figure(figsize=(12, 5))
dendrogram(Z)
plt.title("Single-linkage dendrogram (60 points)")
plt.xlabel("points")
plt.ylabel("merge distance")
plt.show()
```

Read the tree from the bottom up:

- Short vertical joins mean the points were very similar.
- The final merge joins the two crescents at a large distance.
- A horizontal cut just below that final merge gives two clusters.

With single linkage, the tree can chain along a crescent, producing many small
merges at similar heights before the final one. That is the chaining behaviour in
visual form.

Cut the full-data tree at two clusters and compare with the true labels:

```python
for method in ["single", "complete", "average", "ward"]:
    Z = linkage(X_moons, method=method)
    labels = fcluster(Z, t=2, criterion="maxclust")
    print(f"{method}: ARI={adjusted_rand_score(y_moons, labels):.4f}")
```

Expected output:

```text
single: ARI=1.0000
complete: ARI=0.4818
average: ARI=0.4818
ward: ARI=0.4818
```

`fcluster` and `AgglomerativeClustering` agree, as they implement the same merge
sequence. Use whichever API fits the workflow; SciPy gives more control over the cut
height.

## 6. DBSCAN on the Moons

DBSCAN grows clusters through dense regions. It does not need `k`.

```python
for eps in [0.10, 0.15, 0.20, 0.25, 0.30]:
    model = DBSCAN(eps=eps, min_samples=5)
    labels = model.fit_predict(X_moons)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(
        f"eps={eps:.2f}: clusters={n_clusters} noise={n_noise} "
        f"ARI={adjusted_rand_score(y_moons, labels):.4f}"
    )
```

Expected output:

```text
eps=0.10: clusters=4 noise=6 ARI=0.6642
eps=0.15: clusters=2 noise=1 ARI=0.9950
eps=0.20: clusters=2 noise=0 ARI=1.0000
eps=0.25: clusters=2 noise=0 ARI=1.0000
eps=0.30: clusters=2 noise=0 ARI=1.0000
```

At `eps=0.10` the neighbourhood is too small: each crescent breaks into pieces and a
few points become noise. From `0.15` upward, DBSCAN finds the two moons perfectly.
A larger `eps` would eventually merge the moons, so the useful range is bounded on
both sides.

## 7. DBSCAN on Concentric Circles

Circles are impossible for K-means but natural for density-based clustering.

```python
for eps in [0.10, 0.15, 0.20, 0.30]:
    model = DBSCAN(eps=eps, min_samples=5)
    labels = model.fit_predict(X_circles)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(
        f"eps={eps:.2f}: clusters={n_clusters} noise={n_noise} "
        f"ARI={adjusted_rand_score(y_circles, labels):.4f}"
    )
```

Expected output:

```text
eps=0.10: clusters=19 noise=21 ARI=0.5237
eps=0.15: clusters=2 noise=0 ARI=1.0000
eps=0.20: clusters=2 noise=0 ARI=1.0000
eps=0.30: clusters=1 noise=0 ARI=0.0000
```

At `0.15` and `0.20` DBSCAN recovers both rings. At `0.30` the gap between the rings
is bridged and everything merges into one cluster. The parameter is doing real work.

## 8. The Effect of `min_samples`

With `eps` fixed, raise the density threshold.

```python
for min_samples in [3, 5, 10, 20]:
    model = DBSCAN(eps=0.20, min_samples=min_samples)
    labels = model.fit_predict(X_moons)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(
        f"min_samples={min_samples}: clusters={n_clusters} noise={n_noise} "
        f"ARI={adjusted_rand_score(y_moons, labels):.4f}"
    )
```

Expected output:

```text
min_samples=3: clusters=2 noise=0 ARI=1.0000
min_samples=5: clusters=2 noise=0 ARI=1.0000
min_samples=10: clusters=2 noise=1 ARI=0.9950
min_samples=20: clusters=2 noise=2 ARI=0.9901
```

A larger `min_samples` requires denser regions, so more points are labelled noise and
the ARI slips slightly. The change is gradual here because the moons are dense. On
sparse or varying-density data it can be severe.

## 9. Compare the Three Families

```python
rows = []

for name, X, y, model in [
    ("k-means, moons", X_moons, y_moons, KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE)),
    ("single linkage, moons", X_moons, y_moons, AgglomerativeClustering(n_clusters=2, linkage="single")),
    ("ward linkage, moons", X_moons, y_moons, AgglomerativeClustering(n_clusters=2, linkage="ward")),
    ("dbscan, moons", X_moons, y_moons, DBSCAN(eps=0.20, min_samples=5)),
    ("k-means, circles", X_circles, y_circles, KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE)),
    ("dbscan, circles", X_circles, y_circles, DBSCAN(eps=0.15, min_samples=5)),
]:
    labels = model.fit_predict(X)
    rows.append({"method": name, "ARI": adjusted_rand_score(y, labels)})

print(pd.DataFrame(rows).round(4).to_string(index=False))
```

Expected output:

```text
                method     ARI
      k-means, moons  0.2481
single linkage, moons  1.0000
   ward linkage, moons  0.4818
       dbscan, moons  1.0000
    k-means, circles -0.0021
      dbscan, circles  1.0000
```

The table is the day in one view. No method wins everywhere; the assumptions decide.
Single linkage and DBSCAN handle the moons, DBSCAN handles the circles, and K-means
handles neither. On compact, well-separated blobs, K-means would win on speed and
simplicity.

## 10. Clusters as Features Without Leakage

Suppose we want to add a customer segment as a feature for a churn model. The wrong
way fits the clustering on all rows before splitting:

```python
X_all = np.random.default_rng(RANDOM_STATE).normal(size=(500, 2))
y_all = np.random.default_rng(RANDOM_STATE + 1).integers(0, 2, size=500)

segments_all = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE).fit_predict(X_all)
```

Now the test rows have influenced the centroids, which influence the segment labels,
which influence the model. The evaluation will be optimistic.

The right way fits the clustering on the training rows only and applies it to the
rest:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.3, stratify=y_all, random_state=RANDOM_STATE
)

segmenter = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE)
segmenter.fit(X_train)

train_segments = segmenter.predict(X_train)
test_segments = segmenter.predict(X_test)

print("train segment counts:", np.bincount(train_segments))
print("test segment counts:", np.bincount(test_segments))
```

This keeps the transformation honest. DBSCAN has no `predict` method, so applying it
to new data requires either assigning each new point to its nearest core point's
cluster or treating the segment as unknown. Either way, decide the rule on the
training data and document it.

A cluster feature can help a model, but only if it transfers. Re-validate the whole
pipeline, and be prepared for the clusters to drift when the population changes.

## 11. Try It Yourself

1. **Change the noise.** Regenerate the moons with `noise=0.15` and re-run single
   linkage and DBSCAN. Which method degrades first?
2. **Try complete linkage on circles.** Explain why it succeeds or fails.
3. **Plot a k-distance graph.** For each point, compute the distance to its 5th
   nearest neighbor, sort, and look for a knee. Does it suggest `eps=0.15`?
4. **Use a different metric.** Run agglomerative clustering with
   `metric="manhattan"` and compare.
5. **Vary density.** Combine a dense blob and a sparse blob, then run DBSCAN with a
   single `eps`. Explain the failure.
6. **Measure leakage.** Build the same churn model with the wrong and right
   clustering workflows and compare cross-validation scores.

## 12. What We Learned

- K-means fails on moons and circles because its boundaries are straight.
- Single linkage recovered the moons; complete, average, and Ward did not.
- Silhouette preferred the wrong solution on the moons, so metrics need context.
- A dendrogram shows the merge structure and supports cutting at a chosen height.
- DBSCAN recovered both moons and circles by following dense regions.
- `eps` too small fragments clusters; too large merges them.
- `min_samples` controls how much evidence is required to call a region dense.
- Clusters used as features must be fit on training data only.

## Outcome

Learners can apply agglomerative clustering with different linkages, read and cut a
dendrogram, run DBSCAN and interpret clusters versus noise, tune `eps` and
`min_samples`, choose among K-means, hierarchical, and density-based methods, and
use cluster-derived features without contaminating evaluation.
