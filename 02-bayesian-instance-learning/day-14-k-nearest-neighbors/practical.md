# Day 14 — Practical: K-Nearest Neighbors

**Module:** Module 3: Bayesian and Instance-Based Learning

In this practical we build KNN intuition from distances up. We will:

- Compute Euclidean and Manhattan distances by hand.
- Train KNN on Iris and sweep `k`.
- Choose `k` with cross-validation instead of one lucky split.
- Visualize decision boundaries for small and large `k`.
- Compare distance metrics on a curved dataset.
- Compare uniform and distance-weighted voting.
- Show how scaling can make or break KNN.
- Demonstrate the curse of dimensionality with random points.

All datasets are bundled with scikit-learn or generated, so no download is needed.

## 1. Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
```

## 2. Warm-Up: Distances by Hand

Take a query at the origin `q = (0, 0)` and three candidate neighbors.

```python
query = np.array([0.0, 0.0])

points = {
    "A": np.array([3.0, 4.0]),
    "B": np.array([1.0, 1.0]),
    "C": np.array([0.0, 5.0]),
}

for name, point in points.items():
    euclidean = np.linalg.norm(query - point)
    manhattan = np.abs(query - point).sum()
    print(
        f"{name}: euclidean={euclidean:.3f} manhattan={manhattan:.3f}"
    )
```

Expected output:

```text
A: euclidean=5.000 manhattan=7.000
B: euclidean=1.414 manhattan=2.000
C: euclidean=5.000 manhattan=5.000
```

B is the nearest under both metrics. A and C tie under Euclidean distance, but
Manhattan ranks C closer than A. The metric changes the neighbourhood and therefore
the prediction.

With `k = 1`, the prediction is the label of point B. With `k = 3`, all three
points vote, so a 2-to-1 majority decides. That is the entire algorithm.

## 3. Load and Split Iris

```python
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
)

print("train:", X_train.shape, "test:", X_test.shape)
print("classes:", np.bincount(y_train))
```

Expected output:

```text
train: (105, 4) test: (45, 4)
classes: [35 35 35]
```

The classes are balanced, so a dummy classifier that always predicts one class
scores about one third.

```python
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)

print("baseline accuracy:", round(dummy.score(X_test, y_test), 4))
```

Expected output:

```text
baseline accuracy: 0.3333
```

## 4. Sweep `k`

Fit KNN for a range of `k` values inside a scaling pipeline. Record train and test
accuracy.

```python
results = []

for k in [1, 3, 5, 7, 11, 15, 25]:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    model.fit(X_train, y_train)
    results.append(
        {
            "k": k,
            "train": model.score(X_train, y_train),
            "test": model.score(X_test, y_test),
        }
    )

results_df = pd.DataFrame(results).round(4)
print(results_df.to_string(index=False))
```

Expected output:

```text
 k   train    test
 1  1.0000  0.9333
 3  0.9714  0.9111
 5  0.9810  0.9111
 7  0.9810  0.9333
11  0.9714  0.9556
15  0.9810  0.9333
25  0.9143  0.8889
```

`k = 1` has perfect training accuracy and lower test accuracy. As `k` grows,
training accuracy falls because the model smooths over individual points. Test
accuracy on this single split peaks at `k = 11`, but the curve is bumpy.

A single split is noisy. Use cross-validation on the full dataset to choose `k`.

```python
for k in [1, 3, 5, 7, 11, 15, 25]:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    scores = cross_val_score(model, X, y, cv=5)
    print(f"k={k}: mean={scores.mean():.4f}")
```

Expected output:

```text
k=1: mean=0.9467
k=3: mean=0.9533
k=5: mean=0.9600
k=7: mean=0.9600
k=11: mean=0.9467
k=15: mean=0.9400
k=25: mean=0.9400
```

Cross-validation says `k = 5` or `7` is best, not `11`. The earlier peak at 11 was
partly luck. This is a useful lesson: choosing a hyperparameter on a single test
split is a form of overfitting the test set.

## 5. Visualize Decision Boundaries

Project Iris onto petal length and petal width, then draw the boundaries for a small
and a large `k`.

```python
X_2d = X[:, [2, 3]]

def plot_knn_boundary(k, ax):
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    model.fit(X_2d, y)

    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    z = model.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, z, alpha=0.25, cmap="viridis")
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="viridis", edgecolor="k", s=25)
    ax.set_title(f"k = {k}")
    ax.set_xlabel("petal length (cm)")
    ax.set_ylabel("petal width (cm)")


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_knn_boundary(1, axes[0])
plot_knn_boundary(25, axes[1])
plt.tight_layout()
plt.show()
```

The `k = 1` boundary is jagged: it wraps around individual points and creates small
islands. The `k = 25` boundary is smooth but cuts through some genuine structure.
This is the bias-variance trade-off drawn on the plane.

## 6. Distance Metrics on a Curved Dataset

On easy data the metric barely matters. Use two moons to see a difference.

```python
Xm, ym = make_moons(n_samples=400, noise=0.25, random_state=RANDOM_STATE)
Xm_train, Xm_test, ym_train, ym_test = train_test_split(
    Xm, ym, test_size=0.3, stratify=ym, random_state=RANDOM_STATE
)

for metric in ["euclidean", "manhattan", "chebyshev"]:
    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=9, metric=metric),
    )
    model.fit(Xm_train, ym_train)
    print(
        f"{metric}: train={model.score(Xm_train, ym_train):.4f} "
        f"test={model.score(Xm_test, ym_test):.4f}"
    )
```

Expected output:

```text
euclidean: train=0.9357 test=0.9167
manhattan: train=0.9429 test=0.9083
chebyshev: train=0.9500 test=0.9083
```

The differences are small but real. Euclidean wins on this split, while Manhattan
and Chebyshev tie. There is no universal best metric; choose with validation data
and domain knowledge.

## 7. Distance-Weighted Voting

Close neighbors should count more than distant ones. Compare uniform and distance
weights on Iris.

```python
for weights in ["uniform", "distance"]:
    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5, weights=weights),
    )
    model.fit(X_train, y_train)
    print(
        f"{weights}: train={model.score(X_train, y_train):.4f} "
        f"test={model.score(X_test, y_test):.4f}"
    )
```

Expected output:

```text
uniform: train=0.9810 test=0.9111
distance: train=1.0000 test=0.9333
```

Distance weighting raises both scores here. It also makes the boundary less
sensitive to the exact value of `k`, because far neighbors contribute almost
nothing. Do not expect a guaranteed gain on every dataset; validate it.

## 8. Scaling Can Make or Break KNN

Take the breast cancer data and multiply the first feature by 1,000. Now that
feature dominates every distance. Compare KNN with and without standardization.

```python
cancer = load_breast_cancer()
Xc = cancer.data
yc = cancer.target

Xc_scaled_badly = Xc.copy()
Xc_scaled_badly[:, 0] = Xc_scaled_badly[:, 0] * 1000

Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    Xc_scaled_badly, yc, test_size=0.25, stratify=yc, random_state=RANDOM_STATE
)

for use_scaler in [False, True]:
    model = KNeighborsClassifier(n_neighbors=5)
    if use_scaler:
        model = make_pipeline(StandardScaler(), model)
    model.fit(Xb_train, yb_train)
    print(
        f"scaled={use_scaler}: train={model.score(Xb_train, yb_train):.4f} "
        f"test={model.score(Xb_test, yb_test):.4f}"
    )
```

Expected output:

```text
scaled=False: train=0.9249 test=0.9161
scaled=True: train=0.9695 test=0.9790
```

Scaling gains more than six points of test accuracy. Without it, the inflated
feature drowns out the other 29 measurements. Always scale before KNN, and always
inside a pipeline so the scaler is fit on training data only.

## 9. The Curse of Dimensionality

Generate random points in a unit cube for increasing dimensions. For the first
point, compare the nearest and farthest distances to the rest.

```python
rng = np.random.default_rng(RANDOM_STATE)

for d in [1, 2, 5, 10, 50, 100]:
    points = rng.random((500, d))
    distances = np.linalg.norm(points[1:] - points[0], axis=1)
    print(
        f"d={d}: nearest={distances.min():.4f} "
        f"farthest={distances.max():.4f} "
        f"ratio={distances.min() / distances.max():.3f}"
    )
```

Expected output:

```text
d=1: nearest=0.0004 farthest=0.7685 ratio=0.000
d=2: nearest=0.0087 farthest=1.1373 ratio=0.008
d=5: nearest=0.2179 farthest=1.4447 ratio=0.151
d=10: nearest=0.5766 farthest=1.6536 ratio=0.349
d=50: nearest=2.2260 farthest=3.6806 ratio=0.605
d=100: nearest=3.3468 farthest=4.6894 ratio=0.714
```

In one dimension the nearest point is essentially at distance zero while the
farthest is far away. In 100 dimensions the nearest and farthest are much closer to
each other: every point is roughly equally distant. The concept of a "local
neighborhood" has lost its power, which is why KNN degrades in high dimensions.

## 10. Try It Yourself

1. **Weight by inverse square.** Use `weights=lambda d: 1 / (d ** 2)` and compare
   with the built-in distance weighting.
2. **Break `k`.** Set `k = 105` on the Iris training set. What does the model
   predict, and why?
3. **Remove scaling.** Delete the scaler from the Iris pipeline and compare. Iris
   features share units, so the effect is smaller than on breast cancer; explain
   why.
4. **Imbalance experiment.** Drop most versicolor and virginica rows from training
   and see how the majority class dominates. Try `weights="distance"`.
5. **Reduce dimensions.** Apply PCA to breast cancer before KNN and compare accuracy
   and prediction time.
6. **Measure cost.** Time `predict` for 1,000 queries as the training set grows.
   Plot the linear growth.

## 11. What We Learned

- KNN stores the data and predicts from the nearest examples.
- Euclidean, Manhattan, and Minkowski distances define the neighborhood.
- Small `k` has low bias and high variance; large `k` smooths and biases.
- Cross-validation chose `k = 5` or `7`; a single split misleadingly chose `11`.
- Distance weighting gives close neighbors more influence and improved test accuracy.
- Scaling changed breast cancer accuracy from 91.6% to 97.9%.
- In high dimensions, nearest and farthest distances converge, breaking the
  neighborhood assumption.

## Outcome

Learners can compute distances, train KNN with different `k` and metrics, choose
`k` with cross-validation, apply distance weighting and scaling correctly, explain
the curse of dimensionality, and recognize when KNN is the right tool.
