# Day 16 — Theory: Hierarchical Clustering and DBSCAN

**Module:** Module 4: Unsupervised Learning

K-means assumes clusters are compact, roughly spherical, and similar in size. Real
data often refuse those assumptions. Two moons interlock in a crescent shape,
concentric circles wrap around each other, and a few noisy points sit far from
everything. K-means will slice these shapes into halves that cut across the true
structure.

This day adds two alternatives. **Hierarchical clustering** builds a tree of merges
and lets you choose the number of clusters afterwards by cutting the tree.
**DBSCAN** defines clusters as dense regions separated by sparse space, which lets it
find arbitrary shapes and label outliers as noise. Each method makes different
assumptions, and knowing those assumptions is the real skill.

## In Plain English

**Think of it like this:** hierarchical clustering builds a family tree of
points; DBSCAN finds dense clumps separated by empty space.

- **Agglomerative clustering** starts with each point alone and merges the
  closest pairs; the **dendrogram** shows the history.
- You do not have to pick `k` up front — cut the tree where it makes sense.
- **DBSCAN** groups dense regions and marks sparse points as noise.
- DBSCAN finds arbitrary shapes without `k`, but struggles when densities vary.
- Scale features and choose distance and `eps` thoughtfully.

## 1. Beyond K-Means

Consider two interleaving crescents. Any straight partition cuts both crescents in
half, so centroid-based clustering cannot recover them. A method that follows
connectivity or density can.

| Method | Core idea | Shape assumption | Noise handling |
|--------|-----------|------------------|----------------|
| K-means | nearest centroid | spherical, similar size | none; every point assigned |
| Hierarchical | merge similar groups | depends on linkage | none unless cut carefully |
| DBSCAN | dense regions | arbitrary shape | explicitly labels noise |

There is no best method in general. The right choice depends on the shape of the
data, the amount of noise, and what the clusters will be used for.

## 2. Hierarchical Clustering

**Hierarchical clustering** produces a nested sequence of clusterings rather than a
single partition. There are two directions:

- **Agglomerative (bottom-up):** start with every point as its own cluster and
  repeatedly merge the two closest clusters. This is the common approach.
- **Divisive (top-down):** start with one cluster and repeatedly split it. Less
  common because choosing the best split is expensive.

Agglomerative clustering always ends with one cluster containing everything. The
sequence of merges is recorded in a **dendrogram**, and you decide where to cut it.

The algorithm:

1. Compute the distance between every pair of points.
2. Treat each point as a singleton cluster.
3. Merge the two clusters with the smallest **linkage distance**.
4. Recompute distances between the new cluster and the others.
5. Repeat until one cluster remains.

The result does not require choosing `k` in advance. That is the main advantage
over K-means. The cost is that the merge sequence is fixed once made; a later better
merge cannot undo an earlier poor one.

## 3. Linkage Methods

The crucial choice is how to measure the distance between two *clusters*. Four
linkages are common:

**Single linkage:** the distance between the two closest members.

```text
d(A, B) = min_{a in A, b in B} d(a, b)
```

Single linkage can follow long, thin chains, which is why it handles crescents. It
is also prone to **chaining**: a few points bridging two clusters can cause them to
merge.

**Complete linkage:** the distance between the two farthest members.

```text
d(A, B) = max_{a in A, b in B} d(a, b)
```

Complete linkage prefers compact, similarly sized clusters and resists chaining. It
can break elongated clusters.

**Average linkage:** the average distance over all pairs.

```text
d(A, B) = mean_{a in A, b in B} d(a, b)
```

Average linkage is a compromise between single and complete.

**Ward linkage:** the increase in within-cluster variance if two clusters are
merged.

Ward tends to produce compact, spherical clusters similar to K-means. It requires a
Euclidean distance.

Linkage changes the answer dramatically. On two moons, single linkage often recovers
the crescents while complete and Ward split them, because they measure distance
between far edges rather than following the shape.

## 4. Dendrograms

A **dendrogram** is a tree diagram of the merges. The y-axis is the linkage distance
at which two clusters joined; the x-axis lists the points or clusters.

```text
distance
  |
  |        /\
  |       /  \
  |   /\ /    \
  |  /  \      \
  | /    \      \
  +---------------- points
```

Reading a dendrogram:

- **Low joins** mean very similar groups.
- **Tall joins** mean dissimilar groups.
- A horizontal cut at a given height produces a clustering; the number of vertical
  lines crossed is the number of clusters.
- A large vertical gap before the last merge suggests a natural number of clusters.

Cutting the tree is a modelling decision. scikit-learn's `AgglomerativeClustering`
takes `n_clusters`, while SciPy's `fcluster` can cut by height or by a target number
of clusters.

## 5. Distance Metrics

Hierarchical clustering uses a distance matrix, so the metric matters as much as it
does for KNN. Euclidean is the default; Manhattan, cosine, and correlation distances
are also common.

The metric and linkage interact:

- Single linkage with Euclidean can chain through noisy bridges.
- Complete linkage with cosine is common for document clustering.
- Ward requires Euclidean and assumes variance-based geometry.

Always scale features before computing distances. A single large-range feature will
dominate the linkage and produce clusters that reflect only that feature.

## 6. DBSCAN

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) defines a
cluster as a region of high density surrounded by regions of low density. It does
not need `k` and can find arbitrary shapes.

Key definitions, given a radius `eps` and a minimum count `min_samples`:

- **Core point:** a point with at least `min_samples` neighbors within distance
  `eps` (including itself).
- **Border point:** a non-core point within `eps` of a core point.
- **Noise point:** neither core nor border; it belongs to no cluster.

The algorithm:

1. Pick an unvisited point.
2. If it is a core point, start a new cluster and expand it: add all points within
   `eps`, and keep expanding from any core points found.
3. If it is not core, mark it as noise temporarily.
4. Repeat until every point is visited.

Clusters grow through **density-reachable** chains: a point joins if it is close to
a core point, even if it is not core itself. That is what allows DBSCAN to follow
curved shapes.

### The two parameters

- **`eps`** is the neighbourhood radius. Too small and genuine clusters fragment
  into noise; too large and separate clusters merge.
- **`min_samples`** is the density threshold. Larger values require denser regions
  and produce more noise. A common default is `2 * n_features`, with a minimum of 3.

Choosing `eps` is the hard part. A **k-distance plot** helps: sort each point's
distance to its `k`-th nearest neighbor and look for the knee. The knee is a
reasonable `eps`. Domain knowledge often matters more than the plot.

## 7. Density-Based versus Centroid-Based

| Aspect | K-means | DBSCAN |
|--------|---------|--------|
| Number of clusters | must specify `k` | discovered |
| Shapes | spherical | arbitrary |
| Noise | forced into a cluster | labelled as noise |
| Density | assumes similar | threshold-based |
| Large datasets | fast, scalable | can be slow without an index |
| Parameters | `k` | `eps`, `min_samples` |
| Result stability | depends on init | deterministic given parameters |

DBSCAN is a better fit when the data contain irregular shapes or outliers. K-means
is better when clusters are compact and the dataset is large.

## 8. Clustering Before Supervised Learning

Clusters are often used as features for a later model:

- Segment customers, then add the segment id as a feature for churn prediction.
- Cluster locations, then use the cluster as a categorical input.
- Group products, then forecast demand per group.

This can help, but it introduces a serious **leakage** risk. If the clustering is
fit on the full dataset, including the test rows, information from the test set
flows into the features. The model can appear better than it is.

The correct approach is to fit the clustering on the training data only and apply
the learned clusters to validation and test data. For K-means that means storing the
centroids. For DBSCAN there is no `predict` method, so it is harder to apply to new
points; common solutions are to fit on training data and assign new points to the
nearest cluster, or to treat the cluster label as missing for new data.

There is also a conceptual risk: clusters found in one time period or population may
not transfer to another. Treat cluster-derived features like any other learned
transformation and validate them.

## 9. Applications

- **Customer segmentation with irregular profiles:** hierarchical clustering shows
  the merge structure; DBSCAN finds dense groups and isolates unusual customers.
- **Spatial data:** DBSCAN is standard for grouping crime incidents, earthquake
  epicenters, or GPS traces into dense areas while marking isolated events.
- **Anomaly detection:** noise points from DBSCAN are a natural candidate list.
- **Biological taxonomy:** hierarchical clustering mirrors the nested structure of
  species or gene families.
- **Document organization:** hierarchical clustering with cosine distance builds a
  topic tree.

## 10. Strengths and Limitations

**Hierarchical clustering**

- Strengths: no `k` required; produces a full merge tree; deterministic; good for
  small datasets and exploratory analysis.
- Limitations: quadratic or worse memory and time; greedy merges are irreversible;
  sensitive to linkage and distance; no noise handling.

**DBSCAN**

- Strengths: finds arbitrary shapes; discovers the number of clusters; labels noise;
  robust to outliers.
- Limitations: sensitive to `eps` and `min_samples`; struggles with varying density;
  no native prediction for new points; can be slow on large high-dimensional data.

## 11. Common Mistakes

- **Using single linkage and ignoring chaining.** A thin bridge can merge clusters.
- **Forgetting to scale.** Distance-based methods suffer the same bug as KNN.
- **Reading a dendrogram without a cut.** The tree alone does not give a partition.
- **Choosing `eps` arbitrarily.** Use a k-distance plot and validate.
- **Expecting DBSCAN to handle wildly varying densities.** It usually cannot with
  one `eps`.
- **Using DBSCAN on high-dimensional data.** Distances lose meaning.
- **Fitting clustering on all data before a supervised split.** This leaks.
- **Treating noise as a cluster.** Noise is explicitly "unknown", not a group.

## 12. Summary

- Hierarchical clustering builds a merge tree and lets you choose the cut afterwards.
- Single, complete, average, and Ward linkages encode different cluster shapes.
- Dendrograms show the merge distances and support visual cutting.
- DBSCAN finds dense regions of arbitrary shape and labels sparse points as noise.
- `eps` and `min_samples` control DBSCAN's notion of density.
- Cluster-derived features must be fit on training data only to avoid leakage.
- Each method's assumptions matter more than its default settings.

## Outcome

Learners can explain agglomerative clustering and linkage methods, read a dendrogram,
apply DBSCAN and interpret core, border, and noise points, choose between centroid,
hierarchical, and density-based clustering, and use cluster-derived features without
leaking information.
