# Day 14 — Theory: K-Nearest Neighbors

**Module:** Module 3: Bayesian and Instance-Based Learning

Every model so far learned a compact summary: coefficients, a tree, a set of
probabilities, a margin. **K-nearest neighbors (KNN)** does almost the opposite. It
keeps the training data as the model and answers each query by looking at the most
similar stored examples. There is no training phase in the usual sense, no global
formula, and no parameter vector to interpret. The prediction is local, data-driven,
and remarkably intuitive.

That simplicity hides real design choices. What does "nearest" mean? How many
neighbors should vote? Should close neighbors count more than distant ones? And why
does the method degrade so badly in high dimensions? This day answers those
questions and shows where KNN is a sensible tool and where it is not.

## In Plain English

**Think of it like this:** to predict a new case, ask the `k` most similar past
cases and take a vote (classification) or an average (regression).

- KNN **stores** the training data and does its work at prediction time ("lazy
  learning").
- Small `k` is flexible but noisy; large `k` is smooth but may blur boundaries.
- **Distance metric** and **feature scaling** matter enormously.
- It is simple and intuitive, but slow on large data and weak in high
  dimensions.
- A good baseline for small, low-dimensional problems.

## 1. Instance-Based and Lazy Learning

KNN is an **instance-based** learner: it stores instances (training examples) and
uses them directly at prediction time. It is also called **lazy learning** because
it defers computation until a query arrives. Compare it with eager learners:

| Aspect | Eager learner (e.g., logistic regression) | Lazy learner (KNN) |
|--------|-------------------------------------------|--------------------|
| Training | fits parameters | stores the data |
| Prediction | evaluates a formula | searches for neighbors |
| Model size | small, fixed | grows with the training set |
| Interpretability | parameters | the neighbors themselves |
| Adapting to new data | refit | add rows |

The trade-off is clear. Training is essentially free, but every prediction costs a
distance computation to every stored point. For a million rows, that is expensive
unless a spatial index or approximate method is used.

KNN is also **non-parametric**: it does not assume a fixed functional form. The
decision boundary adapts to the data, which lets it fit complex shapes without
feature engineering. The price is that it needs enough data to cover the input
space densely.

## 2. How a Prediction Is Made

For a new point `x`:

1. Compute the distance from `x` to every training point.
2. Sort the distances and take the `k` closest.
3. For classification, let those `k` neighbors vote, and predict the majority
   class. For regression, average their target values.

A small example with `k = 3`:

```text
neighbors:  (distance 0.4, class A)
            (distance 0.7, class B)
            (distance 0.9, class A)

vote: A = 2, B = 1  ->  predict A
```

The prediction is a local average, not a global trend. That is why KNN can separate
classes with irregular boundaries and why it is sensitive to noisy points near the
query.

### Ties

Ties are possible when `k` is even or when two classes receive equal votes. Common
solutions are:

- choose `k` odd for binary problems,
- break ties by distance,
- use `predict_proba` and pick the higher probability.

scikit-learn resolves ties deterministically but the choice is arbitrary. Prefer an
odd `k` for binary classification.

## 3. Distance Functions

The notion of "nearest" depends on a distance metric. Three are common:

**Euclidean distance** (L2):

```text
d(x, z) = sqrt( sum_j (x_j - z_j)^2 )
```

This is the straight-line distance. It is the default and the most intuitive in low
dimensions.

**Manhattan distance** (L1):

```text
d(x, z) = sum_j |x_j - z_j|
```

This is the distance travelled along a grid. It is less sensitive to large
differences in a single feature, which helps with outliers.

**Minkowski distance**:

```text
d(x, z) = ( sum_j |x_j - z_j|^p )^(1/p)
```

`p = 2` gives Euclidean, `p = 1` gives Manhattan, and larger `p` approaches
Chebyshev (maximum coordinate difference). Minkowski is a family, not a single
metric.

The choice of metric is a modelling decision. Euclidean is a reasonable default;
Manhattan can be better for high-dimensional or outlier-heavy data; domain-specific
distances (edit distance, cosine similarity) matter for text and sequences.

## 4. Choosing k

`k` is the single most important hyperparameter, and it directly controls the
bias-variance trade-off.

- **Small `k`** (for example 1): the prediction follows individual points. The
  boundary is jagged, training accuracy is high, and variance is high. One noisy
  point can flip a prediction.
- **Large `k`**: the prediction averages over a wide neighbourhood. The boundary
  smooths out, bias rises, and eventually the model ignores local structure.
- **`k = n`**: every query predicts the global majority class. The model has
  maximum bias and no local information.

The classic picture is a U-shaped validation error curve: error falls as `k` grows,
reaches a minimum, then rises as the model becomes too smooth. Choose `k` by
cross-validation on the training data.

Rules of thumb:

- Start with `k = sqrt(n)` for a rough default.
- Use an odd `k` for binary classification.
- Scale features first, because distance is meaningless otherwise.
- Avoid `k` close to the class count of the smallest class.

### The bias-variance view

With `k = 1`, the model can fit any training set perfectly, so bias is low and
variance is high. As `k` increases, averaging over more neighbors reduces variance
but introduces bias from distant points of other classes. The best `k` balances the
two for the specific data and sample size.

## 5. Distance-Weighted KNN

Not all neighbors are equally informative. A point very close to the query should
count more than one at the edge of the neighborhood. **Distance-weighted KNN** gives
each neighbor a weight inversely related to its distance, often `1 / d`.

```text
weight_i = 1 / distance_i
```

For classification, each class's weight is summed and the largest wins. For
regression, the prediction is the weighted average.

If a neighbor has distance zero (an exact duplicate), the weight becomes infinite.
Implementations handle this by giving that point all the weight, which is a
sensible degenerate rule.

Distance weighting often improves accuracy and smooths the boundary near class
edges. It also reduces the arbitrary effect of the choice of `k`, because far
neighbors contribute little.

## 6. Feature Scaling Is Essential

Distances add up squared differences across features. If one feature ranges from 0
to 1 and another from 0 to 10,000, the second dominates every distance and the first
is effectively ignored.

Standardize each feature before KNN:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier

model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
```

Putting the scaler in a pipeline is also how you avoid leakage during
cross-validation. The scaler must be fit on the training fold only.

Scaling does not change KNN's assumptions; it changes which differences the metric
can see. For features measured in comparable units, scaling may matter less, but it
is almost never harmful.

## 7. The Curse of Dimensionality

KNN works by assuming that nearby points have similar targets. In low dimensions
that is reasonable. In high dimensions it breaks down.

As the number of dimensions grows:

- The volume of the space grows exponentially, so data become sparse.
- The distance to the nearest neighbor and the distance to the farthest neighbor
  become similar. "Nearest" loses meaning.
- To maintain a given density, the number of samples must grow exponentially with
  `d`.

A concrete symptom: draw random points in a unit cube. In one dimension the nearest
neighbor is very close compared with the farthest. In 100 dimensions, the nearest
and farthest distances differ by only a small fraction. Every point is roughly
equally far from every other point, so the neighborhood no longer identifies
similarity.

This is why KNN struggles with images, text, and other high-dimensional inputs
unless the data lie on a lower-dimensional manifold or are first reduced with PCA
or learned embeddings.

## 8. Strengths and Limitations

**Strengths**

- No training phase; trivial to implement and update.
- Non-parametric: can fit complex, irregular boundaries.
- Naturally multiclass.
- Predictions are explainable through the neighbors.
- No assumption about the functional form of the relationship.

**Limitations**

- Prediction is slow for large datasets.
- Memory grows with the training set.
- Extremely sensitive to feature scaling.
- Suffers from the curse of dimensionality.
- Sensitive to irrelevant features and noise.
- Needs a good distance metric, which is domain-specific.
- Imbalanced classes can dominate votes unless weighted.
- No compact model or coefficients to inspect.

## 9. KNN for Regression

For regression, take the average target of the `k` nearest neighbors:

```text
y_hat = (1/k) * sum_i y_i
```

Distance-weighted regression uses a weighted average instead. KNN regression
produces a piecewise-constant or piecewise-smooth fit. It is simple and local, but
it cannot extrapolate beyond the range of the training targets and it struggles
with smooth global trends.

## 10. Real-World Use Case: Recommendation and Similarity Search

KNN's core operation, finding similar items, appears far beyond classification:

- **Recommender systems:** find users similar to you and recommend what they liked.
  In practice, approximate nearest-neighbor indexes make this feasible at scale.
- **Image retrieval:** find visually similar images using embeddings as features.
- **Anomaly detection:** a point far from all its neighbors is unusual.
- **Imputation:** fill a missing value with the average of similar rows.
- **Credit or insurance risk:** find past cases with similar profiles and compare
  outcomes.

In each case the model is only as good as the distance metric and the features. The
algorithm is simple; the engineering is in representation.

## 11. Common Mistakes

- **Forgetting to scale.** The number one KNN bug.
- **Using an even `k` for binary classification.** Ties become arbitrary.
- **Tuning `k` on the test set.** Use cross-validation.
- **Using too many irrelevant features.** They dilute the distance.
- **Ignoring class imbalance.** Majority classes can dominate every neighborhood.
- **Assuming KNN handles high dimensions.** It usually does not.
- **Fitting the scaler on all data.** This leaks information.
- **Using KNN for large-scale inference without an index.** Prediction cost grows
  linearly with the training set.

## 12. Summary

- KNN is a lazy, instance-based learner that stores the training data.
- Predictions use the majority vote or average of the `k` nearest points.
- Euclidean, Manhattan, and Minkowski distances define "nearest".
- Small `k` means low bias and high variance; large `k` means the reverse.
- Distance weighting gives closer neighbors more influence.
- Feature scaling is mandatory, and high dimensions break the method.
- KNN is interpretable and flexible but slow at prediction and weak in high
  dimensions.
- It is the conceptual basis for similarity search and recommendation.

## Outcome

Learners can explain instance-based learning, compute distances with common metrics,
choose `k` with cross-validation, apply distance weighting and scaling, describe the
curse of dimensionality, and decide when KNN is an appropriate model.
