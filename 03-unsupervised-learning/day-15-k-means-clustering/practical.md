# Day 15 — Practical: K-Means Clustering

**Module:** Module 4: Unsupervised Learning

In this practical we run K-means on generated and real data. We will:

- Perform one assignment-update iteration by hand.
- Cluster a four-blob dataset and plot the result.
- Use the elbow method and silhouette score to choose `k`.
- Compare random initialization with k-means++.
- Cluster Iris without using its labels, then compare clusters to species.
- Show why scaling changes customer segments.
- Measure the effect of a single outlier.

All datasets are bundled with scikit-learn or generated, so no download is needed.

## 1. Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

RANDOM_STATE = 42
```

## 2. One Iteration by Hand

Take four points and two initial centroids.

```python
points = np.array([[1.0, 1.0], [2.0, 1.0], [8.0, 8.0], [9.0, 9.0]])
centroids = np.array([[1.0, 1.0], [8.0, 8.0]])

distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
assignments = distances.argmin(axis=1)

print("distances to centroids:")
print(np.round(distances, 3))
print("assignments:", assignments)

new_centroids = np.array(
    [points[assignments == c].mean(axis=0) for c in range(2)]
)
print("new centroids:")
print(new_centroids)
```

Expected output:

```text
distances to centroids:
[[ 0.     9.899]
 [ 1.     9.22 ]
 [ 9.899  0.   ]
 [11.314  1.414]]
assignments: [0 0 1 1]
new centroids:
[[1.5 1. ]
 [8.5 8.5]]
```

Both clusters moved toward their members. Repeating assignment and update until
nothing changes is the whole algorithm.

## 3. Generate Four Blobs

```python
X, y_true = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=0.7,
    random_state=RANDOM_STATE,
)

print("shape:", X.shape)
print("true group sizes:", np.bincount(y_true))
print("feature ranges:", np.round(X.min(axis=0), 1), np.round(X.max(axis=0), 1))
```

Expected output:

```text
shape: (300, 2)
true group sizes: [75 75 75 75]
feature ranges: [-10.6  -8.3] [ 6.3 10.7]
```

The `y_true` labels are available only because we generated the data. In a real
clustering problem they would not exist; we use them here to check the result.

Fit K-means with the correct `k = 4`:

```python
kmeans = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE)
kmeans.fit(X)

print("inertia:", round(kmeans.inertia_, 2))
print("cluster sizes:", np.bincount(kmeans.labels_))
print("iterations:", kmeans.n_iter_)
```

Expected output:

```text
inertia: 277.52
cluster sizes: [75 75 75 75]
iterations: 2
```

The clusters match the true groups perfectly in size, and the algorithm converged in
two iterations from a good initialization.

Plot the result:

```python
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap="viridis", s=25, edgecolor="k")
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=250,
    c="red",
    edgecolor="black",
    label="centroids",
)
plt.legend()
plt.title("K-means with k = 4")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.show()
```

The centroids sit at the centre of each blob. If a centroid looked off to the side,
that would be a hint of a bad initialization or the wrong `k`.

## 4. The Elbow Method

Inertia always falls as `k` rises. Look for the point where the fall slows down.

```python
inertias = []

for k in range(1, 9):
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    model.fit(X)
    inertias.append(model.inertia_)
    print(f"k={k}: inertia={model.inertia_:.1f}")
```

Expected output:

```text
k=1: inertia=19710.6
k=2: inertia=9125.8
k=3: inertia=1840.9
k=4: inertia=277.5
k=5: inertia=251.0
k=6: inertia=225.6
k=7: inertia=200.4
k=8: inertia=178.2
```

The drop from `k=3` (1840.9) to `k=4` (277.5) is enormous. After `k=4` the
improvements are small. The elbow is at 4, which is the true number of blobs.

```python
plt.figure(figsize=(7, 4))
plt.plot(range(1, 9), inertias, marker="o")
plt.xlabel("k")
plt.ylabel("inertia (WCSS)")
plt.title("Elbow method")
plt.show()
```

## 5. The Silhouette Score

Silhouette judges how well each point fits its cluster relative to the nearest other
cluster.

```python
for k in range(2, 9):
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"k={k}: silhouette={score:.4f}")
```

Expected output:

```text
k=2: silhouette=0.6093
k=3: silhouette=0.7888
k=4: silhouette=0.8546
k=5: silhouette=0.7394
k=6: silhouette=0.5993
k=7: silhouette=0.4571
k=8: silhouette=0.3296
```

Silhouette peaks clearly at `k=4`, agreeing with the elbow. When the two methods
agree, the choice is easier. When they disagree, inspect the clusters and use domain
knowledge.

## 6. Initialization Matters

Run K-means once with random initialization and once with k-means++, using a single
restart (`n_init=1`).

```python
for init in ["random", "k-means++"]:
    model = KMeans(
        n_clusters=4, init=init, n_init=1, random_state=RANDOM_STATE
    )
    model.fit(X)
    print(f"{init}: inertia={model.inertia_:.2f}")
```

Expected output:

```text
random: inertia=277.52
k-means++: inertia=277.52
```

On this seed both succeed. Now try several seeds with random initialization:

```python
for seed in [0, 1, 2, 3, 42]:
    model = KMeans(
        n_clusters=4, init="random", n_init=1, random_state=seed
    )
    model.fit(X)
    print(f"random seed={seed}: inertia={model.inertia_:.2f}")
```

Expected output:

```text
random seed=0: inertia=277.52
random seed=1: inertia=8186.37
random seed=2: inertia=1817.43
random seed=3: inertia=1815.72
random seed=42: inertia=277.52
```

Seed 1 reaches an inertia almost thirty times worse. It split a real cluster and
merged parts of others. k-means++ prevents most of these failures:

```python
for seed in [0, 1, 2, 3, 42]:
    model = KMeans(
        n_clusters=4, init="k-means++", n_init=1, random_state=seed
    )
    model.fit(X)
    print(f"k-means++ seed={seed}: inertia={model.inertia_:.2f}")
```

Expected output:

```text
k-means++ seed=0: inertia=277.52
k-means++ seed=1: inertia=277.52
k-means++ seed=2: inertia=277.52
k-means++ seed=3: inertia=277.52
k-means++ seed=42: inertia=277.52
```

Every seed reaches the same good solution. This is why scikit-learn defaults to
k-means++ and multiple restarts. With a small `k`, the extra runs are cheap insurance.

## 7. Cluster Iris Without Labels

Iris has three species, but pretend we do not know them. Standardize the four
features and cluster into three groups.

```python
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)

for k in [2, 3, 4, 5]:
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = model.fit_predict(X_iris_scaled)
    print(
        f"k={k}: inertia={model.inertia_:.2f} "
        f"silhouette={silhouette_score(X_iris_scaled, labels):.4f} "
        f"ARI={adjusted_rand_score(y_iris, labels):.4f}"
    )
```

Expected output:

```text
k=2: inertia=222.36 silhouette=0.5818 ARI=0.5681
k=3: inertia=139.82 silhouette=0.4599 ARI=0.6201
k=4: inertia=114.09 silhouette=0.3869 ARI=0.4728
k=5: inertia=90.93 silhouette=0.3459 ARI=0.4348
```

Silhouette prefers `k=2`, but the true number of species is 3, and the adjusted Rand
index is higher at `k=3`. This is a valuable conflict: an internal metric can
disagree with the structure we care about. Silhouette favours compact, well-separated
groups, and species 1 and 2 overlap in feature space.

Inspect the agreement at `k=3` with a contingency table:

```python
model = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)
labels = model.fit_predict(X_iris_scaled)

print(pd.crosstab(labels, y_iris))
print("adjusted Rand index:", round(adjusted_rand_score(y_iris, labels), 4))
```

Expected output:

```text
col_0   0   1   2
row_0
0       0  39  14
1      50   0   0
2       0  11  36
adjusted Rand index: 0.6201
```

Cluster 1 is exactly setosa. Clusters 0 and 2 split versicolor and virginica, with 14
and 11 points on the wrong side. That is the same overlap we saw in the decision tree
and SVM practicals. The clustering recovered real structure without ever seeing the
labels.

## 8. Scaling Changes Customer Segments

Generate customer data with income in tens of thousands and a spending score from
roughly 0 to 100.

```python
rng = np.random.default_rng(RANDOM_STATE)

income = rng.normal(60000, 15000, 300)
spend = rng.normal(50, 15, 300)
customers = np.c_[income, spend]

scaler_customers = StandardScaler()
customers_scaled = scaler_customers.fit_transform(customers)
```

Cluster without scaling:

```python
raw_model = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)
raw_model.fit(customers)

print("centroids without scaling:")
print(
    pd.DataFrame(raw_model.cluster_centers_, columns=["income", "spend"])
    .round(0)
    .to_string(index=False)
)
```

Expected output:

```text
centroids without scaling:
 income  spend
43089.0   50.0
60697.0   50.0
80042.0   50.0
```

The spending coordinate is essentially 50 for every cluster. Income dominates the
distance so completely that spending is ignored. The algorithm has really produced
three income bands, not three customer types.

Cluster with scaling:

```python
scaled_model = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)
scaled_model.fit(customers_scaled)

centers_original = scaler_customers.inverse_transform(scaled_model.cluster_centers_)

print("centroids with scaling:")
print(
    pd.DataFrame(centers_original, columns=["income", "spend"])
    .round(0)
    .to_string(index=False)
)
```

Expected output:

```text
centroids with scaling:
 income  spend
51084.0   60.0
57465.0   34.0
76539.0   55.0
```

Now spending separates the groups as well. The segments are combinations of income
and behaviour rather than income alone.

Do not compare inertia across the two runs. The scaled version has inertia 265.6,
while the unscaled version has 11,051,715,825.6, but that is a unit change, not an
improvement. Inertia is only comparable on identically scaled data.

## 9. The Effect of One Outlier

Add a single point at `(30, 30)` and compare centroids.

```python
X_with_outlier = np.vstack([X, [[30.0, 30.0]]])

clean_model = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE)
clean_model.fit(X)

outlier_model = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE)
outlier_model.fit(X_with_outlier)

print("clean centroids:")
print(np.round(clean_model.cluster_centers_, 2))
print("centroids with outlier:")
print(np.round(outlier_model.cluster_centers_, 2))
```

Expected output:

```text
clean centroids:
[[-2.62  8.99]
 [-6.85 -6.85]
 [ 4.69  2.02]
 [-8.83  7.23]]
centroids with outlier:
[[-2.62  8.99]
 [-6.85 -6.85]
 [ 5.03  2.39]
 [-8.83  7.23]]
```

One point moved the nearest centroid from `(4.69, 2.02)` to `(5.03, 2.39)` and pulled
it toward the outlier. Three other centroids did not move. With more extreme outliers
or many of them, the distortion grows. K-medoids or DBSCAN are more robust options.

## 10. Try It Yourself

1. **Change `k`.** Run the blob data with `k=3` and `k=5`. Which true groups get
   merged or split?
2. **Use raw features.** Repeat the Iris clustering without standardization. Does
   ARI change? Which features dominate?
3. **Add outliers.** Add ten points at `(30, 30)` and measure how far the centroids
   move.
4. **Try `n_init=1` repeatedly.** Run k-means++ with one restart fifty times and
   record how often it fails to reach inertia 277.52.
5. **Use a non-spherical dataset.** Run K-means on `make_moons` with `k=2` and
   explain why it fails. Compare with DBSCAN tomorrow.
6. **Name the segments.** Take the customer clusters and write a one-sentence
   business description of each, then decide whether the names are plausible.

## 11. What We Learned

- K-means alternates assignment and centroid updates to minimize inertia.
- The elbow and silhouette both pointed to the true `k=4` on the blob data.
- Random initialization failed badly on one seed; k-means++ succeeded on all seeds.
- Iris clustering recovered setosa exactly but split the overlapping species.
- Silhouette preferred `k=2` on Iris while ARI preferred `k=3`; internal metrics are
  not the final word.
- Without scaling, income completely dominated spending.
- Inertia is not comparable across different feature scalings.
- A single outlier can shift a centroid.

## Outcome

Learners can run K-means, interpret centroids and inertia, select `k` with the
elbow method and silhouette score, explain the role of initialization and multiple
restarts, recognize the need for scaling, measure outlier sensitivity, and evaluate
clusters against both geometric metrics and domain meaning.
