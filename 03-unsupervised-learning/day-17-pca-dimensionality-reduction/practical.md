# Day 17 — Practical: PCA and Dimensionality Reduction

**Module:** Module 4: Unsupervised Learning

In this practical we will:

- Load a 30-feature dataset and scale it.
- Fit PCA and read the explained variance.
- Draw a scree plot and a cumulative variance plot.
- Choose a number of components for 95% variance.
- Project the data to two dimensions and plot it.
- Compare a classifier with and without PCA.
- Measure reconstruction error.
- Interpret the first two components.
- Use PCA correctly inside a pipeline to avoid leakage.

All data comes from scikit-learn, so nothing is downloaded.

## 1. Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
```

## 2. Load, Split, and Scale

The breast cancer dataset has 569 rows and 30 numeric features. The target is
malignant versus benign. We split first, then fit the scaler on the training rows.

```python
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("train:", X_train.shape, "test:", X_test.shape)
```

Expected output:

```text
train: (426, 30) test: (143, 30)
```

Standardization is essential here: the features include area (hundreds), radius
(tens), and smoothness (fractions). Without scaling, area would dominate every
component.

## 3. Fit PCA and Read Explained Variance

```python
pca = PCA().fit(X_train_s)

print("components:", pca.n_components_)
print("first 10 ratios:", np.round(pca.explained_variance_ratio_[:10], 4))
print("first 2 total:", round(pca.explained_variance_ratio_[:2].sum(), 4))
```

Expected output:

```text
components: 30
first 10 ratios: [0.439  0.1952 0.0941 0.0679 0.0558 0.0401 0.0219 0.0157 0.0124
 0.0116]
first 2 total: 0.6341
```

The first component alone captures 43.9% of the variance, and the first two capture
63.4%. The last 20 components together carry a small fraction. That imbalance is
what makes compression possible.

## 4. Scree Plot and Cumulative Variance

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(np.arange(1, 31), pca.explained_variance_ratio_, "o-")
axes[0].set_xlabel("component")
axes[0].set_ylabel("explained variance ratio")
axes[0].set_title("scree plot")

cumulative = np.cumsum(pca.explained_variance_ratio_)
axes[1].plot(np.arange(1, 31), cumulative, "o-")
axes[1].axhline(0.95, color="red", linestyle="--", label="95%")
axes[1].set_xlabel("number of components")
axes[1].set_ylabel("cumulative variance")
axes[1].set_title("cumulative explained variance")
axes[1].legend()

plt.tight_layout()
plt.show()
```

Expected cumulative ratios for the first ten components:

```text
[0.439  0.6341 0.7282 0.7961 0.8519 0.892  0.9139 0.9295 0.942  0.9536]
```

The curve rises steeply for the first few components and then flattens. There is no
dramatic elbow, but keeping around 10 components gets most of the variance.

## 5. How Many Components?

```python
cumulative = np.cumsum(pca.explained_variance_ratio_)

for target in [0.90, 0.95, 0.99]:
    n = np.searchsorted(cumulative, target) + 1
    print(f"{target:.0%} variance needs {n} components")
```

Expected output:

```text
90% variance needs 7 components
95% variance needs 10 components
99% variance needs 17 components
```

Thirty features reduce to ten if we accept 95% variance, a 3x compression. Whether
that is good depends on the task, which we check next.

## 6. Visualize in Two Dimensions

```python
Z = pca.transform(X_test_s)

plt.figure(figsize=(7, 5))
for label, name, color in [(0, "malignant", "tab:red"), (1, "benign", "tab:blue")]:
    mask = y_test == label
    plt.scatter(Z[mask, 0], Z[mask, 1], s=20, alpha=0.7, label=name, color=color)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("breast cancer projected onto first two components")
plt.legend()
plt.show()
```

The two classes separate well along PC1, with some overlap near the middle. A 2D plot
is enough to see the structure, even though it keeps only 63.4% of the variance. That
does not prove a 2-component model is sufficient, but it is a useful summary.

## 7. Downstream Model: Original versus PCA

Train the same classifier on different representations.

```python
def evaluate(transform):
    steps = [StandardScaler()]
    if transform is not None:
        steps.append(PCA(n_components=transform))
    steps.append(LogisticRegression(max_iter=1000))
    model = make_pipeline(*steps)
    model.fit(X_train, y_train)
    return model.score(X_train, y_train), model.score(X_test, y_test)

for name, transform in [
    ("original (30)", None),
    ("pca 2", 2),
    ("pca 5", 5),
    ("pca 10", 10),
    ("pca 95%", 0.95),
]:
    train_acc, test_acc = evaluate(transform)
    print(f"{name}: train={train_acc:.4f} test={test_acc:.4f}")
```

Expected output:

```text
original (30): train=0.9883 test=0.9860
pca 2: train=0.9624 test=0.9441
pca 5: train=0.9789 test=0.9650
pca 10: train=0.9883 test=0.9790
pca 95%: train=0.9883 test=0.9790
```

Two components lose about 4 points of accuracy. Five lose about 2. Ten components are
within 1 point of the full model, with a third of the features. For this dataset PCA
is a reasonable compression, but it is not free: the drop is real.

Note that `n_components=0.95` keeps 10 components, matching our manual choice.

## 8. Reconstruction Error

Map the reduced data back to 30 dimensions and measure the error in standardized
units.

```python
for k in [2, 5, 10, 20, 30]:
    p = PCA(n_components=k).fit(X_train_s)
    Z_test = p.transform(X_test_s)
    reconstructed = p.inverse_transform(Z_test)
    mse = np.mean((X_test_s - reconstructed) ** 2)
    print(f"k={k:2d}: reconstruction MSE={mse:.4f}")
```

Expected output:

```text
k= 2: reconstruction MSE=0.3223
k= 5: reconstruction MSE=0.1489
k=10: reconstruction MSE=0.0516
k=20: reconstruction MSE=0.0060
k=30: reconstruction MSE=0.0000
```

The error falls quickly and then slowly. At 10 components the average squared error
per standardized feature is about 0.05. At 30 it is zero, because nothing is
discarded. This curve is another way to choose `k`, though it measures reconstruction
rather than task performance.

## 9. What Do the Components Mean?

Look at the largest loadings of the first two components.

```python
pc1 = pd.Series(pca.components_[0], index=cancer.feature_names)
pc2 = pd.Series(pca.components_[1], index=cancer.feature_names)

print("PC1 largest loadings:")
print(pc1[pc1.abs().sort_values(ascending=False).head(6).index].round(3))

print("\nPC2 largest loadings:")
print(pc2[pc2.abs().sort_values(ascending=False).head(6).index].round(3))
```

Expected output:

```text
PC1 largest loadings:
mean concave points    0.263
mean concavity         0.258
worst concave points   0.252
mean compactness       0.242
worst perimeter        0.237
worst concavity        0.232

PC2 largest loadings:
mean fractal dimension     0.363
fractal dimension error    0.278
worst fractal dimension    0.277
compactness error          0.238
mean radius               -0.233
mean area                 -0.229
```

PC1 is dominated by concavity and compactness measures, all with the same sign. It
reads as an overall "irregularity" axis: larger values mean more concave, less
compact cells. PC2 contrasts fractal-dimension measures with radius and area, a
different kind of shape-versus-size contrast.

These interpretations are approximate. The components mix many features, and the
signs are arbitrary (multiplying a component by -1 changes nothing). Do not over-read
them.

## 10. PCA with a Different Classifier

PCA is not tied to logistic regression.

```python
for k in [2, 5, 10, 20, 30]:
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=k),
        KNeighborsClassifier(n_neighbors=5),
    ).fit(X_train, y_train)
    print(f"knn, k={k:2d}: test={model.score(X_test, y_test):.4f}")
```

Expected output:

```text
knn, k= 2: test=0.9371
knn, k= 5: test=0.9441
knn, k=10: test=0.9790
knn, k=20: test=0.9860
knn, k=30: test=0.9790
```

KNN also benefits from dimensionality reduction up to a point. At 20 components it
matches the full 30-feature result, and at 30 it is slightly worse. The noise removed
by dropping a few low-variance components can help a distance-based model.

## 11. PCA on Iris for Visualization

Iris is small enough to plot directly, but it is a clean demonstration.

```python
iris = load_iris()
X_iris = StandardScaler().fit_transform(iris.data)

pca_iris = PCA(n_components=2).fit(X_iris)
Z_iris = pca_iris.transform(X_iris)

print("iris explained variance:", np.round(pca_iris.explained_variance_ratio_, 4))

plt.figure(figsize=(7, 5))
for label, name in enumerate(iris.target_names):
    mask = iris.target == label
    plt.scatter(Z_iris[mask, 0], Z_iris[mask, 1], s=25, alpha=0.8, label=name)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("iris projected onto first two components")
plt.legend()
plt.show()
```

Expected output:

```text
iris explained variance: [0.7296 0.2285]
```

Two components retain 95.8% of the variance. Setosa separates cleanly; versicolor
and virginica overlap a little, which matches what a classifier finds.

## 12. Fit PCA in a Pipeline to Avoid Leakage

The safe pattern is to put scaling and PCA inside a pipeline so that each
cross-validation fold fits them on its own training part.

```python
from sklearn.model_selection import cross_val_score

safe_model = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.95),
    LogisticRegression(max_iter=1000),
)

scores = cross_val_score(safe_model, X, y, cv=5, scoring="accuracy")
print("fold scores:", np.round(scores, 4))
print("mean:", round(scores.mean(), 4))
```

Expected output:

```text
fold scores: [0.9912 0.9737 0.9825 0.9737 0.9823]
mean: 0.9807
```

The unsafe alternative fits PCA on all rows before splitting. That leaks the test
distribution into the components, so scores look better than they are. The pipeline
prevents this automatically.

## 13. Try It Yourself

1. **Run PCA without scaling.** Which feature dominates PC1? Explain the result.
2. **Keep 99% variance.** How many components are needed on breast cancer, and how
   does the test accuracy change?
3. **Add noise features.** Append 50 random columns, rerun PCA, and check whether the
   first components are dominated by noise.
4. **Use a different scaler.** Try `MinMaxScaler` and compare the explained variance.
5. **Detect anomalies.** Flag the test rows with the highest reconstruction error at
   `k=5`. Are they unusual in the original features?
6. **Try a nonlinear method.** Apply t-SNE to the same data for visualization and
   compare with the PCA plot. (t-SNE is in scikit-learn.)

## 14. What We Learned

- PCA finds orthogonal directions of maximum variance.
- Standardizing before PCA is essential for mixed-unit features.
- Explained variance and cumulative curves guide the choice of `k`.
- On breast cancer, 10 components retain 95% of the variance and nearly all the
  accuracy.
- Reconstruction error measures how much information was discarded.
- Components are mixtures, so interpret them cautiously.
- PCA helps KNN and logistic regression but is not guaranteed to help every model.
- Scaling and PCA must be fit inside the training fold or leakage results.

## Outcome

Learners can fit and interpret PCA, read scree and cumulative variance plots, choose
a component count for a target variance, measure reconstruction error, project data
for visualization, compare models with and without PCA, interpret loadings, and use
PCA inside a leak-free pipeline.
