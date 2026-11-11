# Day 15 — Theory: Clustering and K-Means

**Module:** Module 4: Unsupervised Learning

Every problem so far came with labels. We knew which emails were spam, which tumors
were malignant, and which flower was which species. In many real situations nobody
provides the answer. A marketing team has customer records but no pre-defined
segments. A biologist has gene expression measurements but no known cell types. An
operations team has sensor readings but no list of machine states.

**Unsupervised learning** looks for structure without a target. **Clustering** is
the most common form: group rows so that members of a group are more similar to each
other than to rows outside it. **K-means** is the simplest and most widely used
clustering algorithm, and it is a good entry point into both the power and the
pitfalls of unsupervised methods.

## In Plain English

**Think of it like this:** K-means drops `k` magnets into the data, lets each
point stick to the nearest magnet, then moves each magnet to the middle of its
group and repeats.

- The loop is: **assign points → move centers → repeat** until stable.
- It minimizes the distance inside clusters, so clusters come out round and
  similar in size.
- Choose `k` with the **elbow method** or **silhouette score**.
- Always scale features first, and run several times with different starts.
- It struggles with odd-shaped clusters and outliers.

## 1. Supervised versus Unsupervised Learning

| Aspect | Supervised | Unsupervised |
|--------|------------|--------------|
| Data | features plus labels | features only |
| Goal | predict a target | find structure |
| Evaluation | compare to known answers | internal measures, domain judgement |
| Typical tasks | regression, classification | clustering, dimensionality reduction |
| Feedback | error against label | no direct error |

Without labels there is no single "correct" answer to compare against. A clustering
can be useful for one business question and meaningless for another. This makes
evaluation harder and makes domain knowledge essential. A high silhouette score does
not guarantee that the clusters mean anything.

## 2. What Clustering Is Used For

Clustering supports many tasks:

- **Customer segmentation:** group customers by behaviour so marketing can target
  each group differently.
- **Anomaly detection:** points that do not belong to any dense cluster are
  suspicious.
- **Image compression:** replace many similar colours with one representative.
- **Document grouping:** organize news articles or support tickets by topic.
- **Biological discovery:** group cells, genes, or patients by similarity.
- **Data exploration:** find structure before building a supervised model.

In each case the clusters are a *tool*, not the final answer. They need names,
validation, and a decision about what to do with each group.

## 3. The K-Means Algorithm

K-means partitions `n` points into `k` groups. It represents each group by its
**centroid**, the mean of the points assigned to it.

The algorithm alternates two steps:

1. **Assignment:** assign each point to the nearest centroid.
2. **Update:** move each centroid to the mean of the points assigned to it.

Repeat until assignments stop changing (or a maximum number of iterations is
reached).

```text
choose k initial centroids
repeat:
    for each point:
        assign it to the closest centroid
    for each cluster:
        centroid = mean of its assigned points
until assignments do not change
```

A single iteration on a small example:

```text
points:  (1,1) (2,1) (8,8) (9,9)
initial centroids:  (1,1) and (8,8)

assign:
    (1,1) -> c1, (2,1) -> c1
    (8,8) -> c2, (9,9) -> c2

update:
    c1 = mean of (1,1),(2,1) = (1.5, 1.0)
    c2 = mean of (8,8),(9,9) = (8.5, 8.5)
```

The next assignment uses the new centroids. If no point changes cluster, the
algorithm stops. Because both steps never increase the objective, K-means is
guaranteed to converge, though not necessarily to the global optimum.

### Hard assignments

Each point belongs to exactly one cluster. That is a **hard** assignment. A
probabilistic variant, Gaussian mixture models, gives soft assignments, where a
point has a probability of belonging to each cluster. K-means is the simpler,
faster cousin.

## 4. The Objective: Within-Cluster Sum of Squares

K-means minimizes the **within-cluster sum of squares (WCSS)**, also called
**inertia**:

```text
J = sum_{clusters c} sum_{points i in c} ||x_i - mu_c||^2
```

Each term is the squared distance from a point to its centroid. The objective is
low when points are close to their own centroids.

For a fixed assignment, the mean is the point that minimizes the sum of squared
distances, which is why the update step uses the mean. For fixed centroids, the
nearest centroid minimizes each point's term, which is why the assignment step is
optimal. Alternating the two steps therefore never increases `J`.

The objective has a scale: doubling the units quadruples inertia. It is useful for
comparing runs on the same data, not for comparing across datasets or feature
scales.

### Local minima

`J` is not convex in the joint assignment, so K-means can converge to a local
minimum. Different initial centroids can lead to different solutions with different
inertia. The standard remedy is to run the algorithm several times with different
initializations and keep the best result (`n_init` in scikit-learn).

## 5. Choosing k

`k` must be chosen before running the algorithm, and it is not learned from the
data. Two common tools help.

### The elbow method

Plot inertia against `k`. Inertia always decreases as `k` grows, reaching zero when
every point is its own cluster. We look for an **elbow**: the point where adding
another cluster stops producing a large drop.

```text
inertia
  |\
  | \
  |  \___
  |      \____
  |           \______
  +------------------- k
        4
```

The elbow is a heuristic. On smooth or overlapping data there may be no clear bend,
and the choice becomes a judgement call.

### The silhouette score

For each point `i`, let `a(i)` be its average distance to points in its own cluster
and `b(i)` its average distance to points in the nearest other cluster. The
**silhouette** of the point is:

```text
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

It ranges from -1 to 1:

- near 1: the point is much closer to its own cluster than to the nearest other,
- near 0: the point sits between clusters,
- negative: the point may be in the wrong cluster.

The overall silhouette is the average over all points. A higher score generally
means better-separated clusters, but it tends to favour convex, compact clusters,
which is exactly what K-means produces. Do not use it blindly.

Other tools include the Calinski-Harabasz index and the Davies-Bouldin index. All
are internal measures: they judge geometry, not usefulness.

## 6. Initialization and k-means++

The initial centroids matter. Bad initialization can split a natural cluster or
merge two of them, producing a poor local minimum.

**Random initialization** picks `k` random points or coordinates. It can be
unlucky, especially with many clusters.

**k-means++** chooses the first centroid at random, then picks each subsequent
centroid with probability proportional to its squared distance from the nearest
already-chosen centroid. This spreads the starting points out and dramatically
reduces bad runs.

scikit-learn defaults to `init="k-means++"` and `n_init=10`, meaning it runs ten
independent initializations and keeps the best inertia. Setting `n_init=1` is
faster but less reliable; the practical will show a case where random
initialization gets stuck.

## 7. Assumptions and Limitations

K-means implicitly assumes:

- **Clusters are roughly spherical.** The objective penalizes squared distance in
  all directions equally, so elongated clusters are split or merged.
- **Clusters have similar size.** A large cluster can pull a centroid away from a
  small one.
- **Clusters have similar density.** Dense regions dominate the mean.
- **`k` is known.** The algorithm cannot discover the number of clusters.
- **Features are scaled comparably.** Otherwise one feature dominates the distance.

It is also:

- **Sensitive to outliers**, because a mean moves toward extreme points.
- **Sensitive to initialization**, mitigated by multiple restarts.
- **Not suitable for arbitrary shapes**, such as moons or rings, unless the data
  are transformed.

For irregular shapes and noise, density-based methods such as DBSCAN are usually
better. That is the next day.

## 8. Scaling and Outliers

Distance-based methods require comparable units. If income ranges over tens of
thousands and a spending score over 0 to 100, income dominates every distance and
spending is effectively ignored. Standardize features first, and fit the scaler on
the data used for clustering.

Outliers are a separate problem. A single extreme point can shift a centroid enough
to steal points from a real cluster. Options include:

- inspecting and removing obvious outliers,
- using a robust variant such as K-medoids, which uses actual points as centers,
- using DBSCAN, which labels low-density points as noise.

## 9. Evaluating Clusters Without Labels

When labels exist, adjusted Rand index (ARI) or normalized mutual information can
compare clusters with truth. Usually there are no labels, so we rely on:

- **internal measures** such as silhouette, Calinski-Harabasz, and Davies-Bouldin,
- **stability** across random seeds or resamples,
- **domain review**: do the groups make sense to the people who will use them?

A cluster that is statistically tidy but business-meaningless is not useful. The
best evaluation combines geometry with human judgement.

## 10. Real-World Use Case: Customer Segmentation

A retailer has records of annual income and a spending score for each customer. The
goal is to design different marketing campaigns.

- **Features:** income, spending score, recency, frequency, and average basket size.
- **Scaling:** standardize all features.
- **`k`:** try 2 to 10, use the elbow and silhouette, then talk to marketing.
- **Result:** for example, budget-conscious low spenders, high-income low spenders,
  and high-income high spenders.
- **Action:** each segment receives a different offer and channel.
- **Caution:** clusters describe correlation, not causation. A segment does not
  tell you that changing income would change spending.

The value is in the decisions the segments enable, not in the inertia value.

## 11. Common Mistakes

- **Forgetting to scale.** The most common K-means bug.
- **Choosing `k` from a single metric without inspection.** Plot the clusters.
- **Using `n_init=1`.** You will sometimes get a bad local minimum.
- **Treating clusters as ground truth.** They are a summary, not a discovery of
  natural kinds.
- **Using K-means on non-spherical shapes.** Try DBSCAN or a mixture model.
- **Ignoring outliers.** One extreme point can distort a centroid.
- **Comparing inertia across datasets or scalings.** It is not comparable.
- **Skipping the domain conversation.** Clusters need names and owners.

## 12. Summary

- Unsupervised learning finds structure without labels.
- K-means alternates assignment and centroid updates to minimize inertia.
- The objective is WCSS, and the algorithm converges to a local minimum.
- `k` is a hyperparameter, chosen with the elbow method, silhouette, and judgement.
- k-means++ initialization and multiple restarts avoid bad solutions.
- K-means assumes spherical, similar-sized, similarly dense clusters and scaled
  features.
- Outliers and unequal feature ranges distort the result.
- Clusters need domain interpretation and are not causal explanations.

## Outcome

Learners can explain clustering and unsupervised learning, describe the K-means
assignment-update loop, write the WCSS objective, choose `k` with the elbow method
and silhouette score, explain initialization and the need for multiple restarts,
identify K-means assumptions and failure modes, and interpret clusters responsibly.
