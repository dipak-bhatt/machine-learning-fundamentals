# Day 10 — Practical: Support Vector Machines

**Module:** Module 2: Supervised Learning

In this practical we build intuition for margins and kernels, then use SVMs on real
data. We will:

- Compute the distance from a point to a hyperplane by hand.
- Train a linear SVM on separable data and identify its support vectors.
- Watch `C` change the margin and the number of support vectors.
- Compare linear, polynomial, and RBF kernels on a curved dataset.
- See how `gamma` controls the flexibility of an RBF boundary.
- Demonstrate why feature scaling is mandatory.
- Compare SVMs with logistic regression.
- Tune an RBF SVM on the breast cancer data with cross-validation.

Nothing is downloaded; every dataset is generated or bundled with scikit-learn.

## 1. Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

RANDOM_STATE = 42
```

`RANDOM_STATE` fixes the generated data and the train/test splits. SVM training
itself is deterministic for a fixed `C` and kernel.

## 2. Warm-Up: Distance to a Hyperplane

Take the hyperplane `w . x + b = 0` with `w = [1, 1]` and `b = -2`. Compute the
signed score and the distance for the point `x = [2, 3]`.

```python
w = np.array([1.0, 1.0])
b = -2.0
x = np.array([2.0, 3.0])

score = w @ x + b
distance = abs(score) / np.linalg.norm(w)

print("score:", score)
print("||w||:", round(np.linalg.norm(w), 3))
print("distance:", round(distance, 3))
```

Expected output:

```text
score: 3.0
||w||: 1.414
distance: 2.121
```

The score is positive, so the point is on the positive side. Its distance is
`2.121` units. The margin band width is `2 / ||w|| = 1.414`, which is the quantity
the SVM tries to maximize by shrinking `||w||`.

## 3. Generate Separable Data

Two Gaussian blobs make a clean binary problem that a line can separate.

```python
X, y = make_blobs(
    n_samples=200,
    centers=2,
    cluster_std=0.8,
    random_state=RANDOM_STATE,
)

print("shape:", X.shape)
print("class counts:", np.bincount(y))
print("feature ranges:", np.round(X.min(axis=0), 1), np.round(X.max(axis=0), 1))
```

Expected output:

```text
shape: (200, 2)
class counts: [100 100]
feature ranges: [-4.6  0.4] [ 6.5 11.2]
```

Split, then fit a linear SVM inside a scaling pipeline:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
)

svm = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))
svm.fit(X_train, y_train)

print("train accuracy:", round(svm.score(X_train, y_train), 3))
print("test accuracy:", round(svm.score(X_test, y_test), 3))
```

Expected output:

```text
train accuracy: 1.0
test accuracy: 1.0
```

Both classes are cleanly separated, so the linear SVM achieves perfect accuracy.

## 4. Support Vectors and the Margin

Only the points on the margin band determine the boundary. Count them:

```python
svc = svm.named_steps["svc"]

print("support vectors per class:", svc.n_support_)
print("total support vectors:", svc.n_support_.sum())
print("support vector shape:", svc.support_vectors_.shape)
```

Expected output:

```text
support vectors per class: [1 2]
total support vectors: 3
support vector shape: (3, 2)
```

Out of 140 training points, only 3 are support vectors. Remove the other 137 and
the boundary would not move. This is the practical meaning of the margin: almost
all the data is irrelevant once the closest points are identified.

Visualize the decision boundary and circle the support vectors:

```python
def plot_boundary(model, X, y, ax=None):
    ax = ax or plt.gca()
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    z = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, z, alpha=0.2, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=20)
    return ax


ax = plot_boundary(svm, X, y)
scaler = svm.named_steps["standardscaler"]
support_original = scaler.inverse_transform(svc.support_vectors_)
ax.scatter(
    support_original[:, 0],
    support_original[:, 1],
    s=180,
    facecolors="none",
    edgecolors="black",
    linewidths=1.5,
    label="support vectors",
)
ax.legend()
plt.show()
```

The circled points sit on the edge of the margin band. The band itself is the
widest corridor that fits between the classes.

## 5. The Effect of `C`

`C` controls how strongly the model punishes points inside the margin or on the
wrong side. Use a noisier dataset so the trade-off is visible.

```python
Xm, ym = make_moons(n_samples=400, noise=0.25, random_state=RANDOM_STATE)
Xm_train, Xm_test, ym_train, ym_test = train_test_split(
    Xm, ym, test_size=0.3, stratify=ym, random_state=RANDOM_STATE
)

for C in [0.01, 0.1, 1, 10, 100, 1000]:
    model = make_pipeline(
        StandardScaler(), SVC(kernel="rbf", C=C, gamma="scale")
    )
    model.fit(Xm_train, ym_train)
    n_sv = model.named_steps["svc"].n_support_.sum()
    print(
        f"C={C}: train={model.score(Xm_train, ym_train):.3f} "
        f"test={model.score(Xm_test, ym_test):.3f} support={n_sv}"
    )
```

Expected output:

```text
C=0.01: train=0.821 test=0.925 support=280
C=0.1: train=0.886 test=0.942 support=160
C=1: train=0.921 test=0.942 support=93
C=10: train=0.943 test=0.917 support=55
C=100: train=0.964 test=0.908 support=43
C=1000: train=0.968 test=0.892 support=37
```

Read the table from left to right:

- Very small `C` allows many violations, keeps a wide margin, and has many support
  vectors. Training accuracy is low but the model is smooth.
- Moderate `C` balances the two and gives the best test score.
- Very large `C` nearly memorizes the training set, uses few support vectors, and
  overfits. Test accuracy falls even as training accuracy climbs.

The number of support vectors is a useful diagnostic: it usually shrinks as `C`
grows, because the margin tightens around fewer decisive points.

## 6. Kernels on Curved Data

The two moons dataset cannot be separated by a straight line. Compare three
kernels on the same split.

```python
for kernel in ["linear", "poly", "rbf"]:
    model = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=1.0))
    model.fit(Xm_train, ym_train)
    print(
        f"{kernel}: train={model.score(Xm_train, ym_train):.3f} "
        f"test={model.score(Xm_test, ym_test):.3f}"
    )
```

Expected output:

```text
linear: train=0.818 test=0.892
poly: train=0.807 test=0.867
rbf: train=0.921 test=0.942
```

The linear kernel is limited by its straight boundary. The polynomial kernel with
the default degree does no better here. The RBF kernel captures the curved shape
and gains about five points of test accuracy.

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, kernel in zip(axes, ["linear", "poly", "rbf"]):
    model = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=1.0))
    model.fit(Xm_train, ym_train)
    plot_boundary(model, Xm, ym, ax=ax)
    ax.set_title(f"{kernel} kernel")

plt.tight_layout()
plt.show()
```

The three panels make the difference visible: a straight line, a mildly curved
boundary, and a flexible boundary that follows the moons.

## 7. The Effect of `gamma`

For the RBF kernel, `gamma` sets how quickly influence fades with distance. Fix
`C=1` and sweep `gamma`.

```python
for gamma in [0.01, 0.1, 1.0, 10.0, 100.0]:
    model = make_pipeline(
        StandardScaler(), SVC(kernel="rbf", C=1.0, gamma=gamma)
    )
    model.fit(Xm_train, ym_train)
    print(
        f"gamma={gamma}: train={model.score(Xm_train, ym_train):.3f} "
        f"test={model.score(Xm_test, ym_test):.3f}"
    )
```

Expected output:

```text
gamma=0.01: train=0.811 test=0.892
gamma=0.1: train=0.821 test=0.925
gamma=1.0: train=0.946 test=0.917
gamma=10.0: train=0.964 test=0.908
gamma=100.0: train=0.993 test=0.883
```

- Small `gamma` makes each point's influence broad, producing a smooth boundary
  that underfits.
- Moderate `gamma` follows the shape of the data.
- Large `gamma` makes each point influential only very locally, producing islands
  around individual training points and overfitting.

`gamma="scale"`, the scikit-learn default, sets `gamma = 1 / (p * X.var())` and is
a reasonable starting point. Tune it together with `C`.

## 8. Why Scaling Matters

Take the moons data and multiply the second feature by 100. The distances now
compare a value near 1 with a value near 100, so the second feature dominates the
RBF similarity.

```python
X_scaled_badly = Xm.copy()
X_scaled_badly[:, 1] = X_scaled_badly[:, 1] * 100

Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_scaled_badly, ym, test_size=0.3, stratify=ym, random_state=RANDOM_STATE
)

for use_scaler in [False, True]:
    model = SVC(kernel="rbf", C=1.0, gamma="scale")
    if use_scaler:
        model = make_pipeline(StandardScaler(), model)
    model.fit(Xb_train, yb_train)
    print(
        f"scaled={use_scaler}: train={model.score(Xb_train, yb_train):.3f} "
        f"test={model.score(Xb_test, yb_test):.3f}"
    )
```

Expected output:

```text
scaled=False: train=0.775 test=0.875
scaled=True: train=0.921 test=0.942
```

Without scaling, the model wastes its capacity on the inflated feature and loses
almost seven points of test accuracy. With scaling, the two features contribute
fairly. Always put `StandardScaler` inside the pipeline so it is fit on training
folds only.

## 9. Compare with Logistic Regression

SVMs are not the only margin-like classifier. Compare the two linear models and the
RBF SVM on the same moons split.

```python
models = {
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "linear SVM": make_pipeline(StandardScaler(), SVC(kernel="linear")),
    "RBF SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale")),
}

for name, model in models.items():
    model.fit(Xm_train, ym_train)
    print(
        f"{name}: train={model.score(Xm_train, ym_train):.3f} "
        f"test={model.score(Xm_test, ym_test):.3f}"
    )
```

Expected output:

```text
logistic: train=0.814 test=0.900
linear SVM: train=0.818 test=0.892
RBF SVM: train=0.921 test=0.942
```

Logistic regression and the linear SVM are close, which is expected: both draw a
single line, and on this split they find similar ones. The nonlinear RBF SVM wins
because the true boundary is curved. If the data were linearly separable, the three
would converge to similar accuracy.

## 10. Real Data: Breast Cancer Classification

The breast cancer dataset has 569 rows and 30 numeric features. It is a good test
of scaling and tuning on a realistic binary problem.

```python
cancer = load_breast_cancer()
Xc = cancer.data
yc = cancer.target

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.25, stratify=yc, random_state=RANDOM_STATE
)

print("train/test shapes:", Xc_train.shape, Xc_test.shape)
print("class counts:", np.bincount(yc_train), np.bincount(yc_test))
print("feature range:", round(Xc.min(), 2), "to", round(Xc.max(), 2))
```

Expected output:

```text
train/test shapes: (426, 30) (143, 30)
class counts: [159 267] [53 90]
feature range: 0.0 to 4254.0
```

The feature range spans more than four thousand units, so scaling is essential.
Compare four models on the same split:

```python
models = {
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "linear SVM": make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0)),
    "RBF SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")),
    "LinearSVC": make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=5000)),
}

for name, model in models.items():
    model.fit(Xc_train, yc_train)
    print(
        f"{name}: train={model.score(Xc_train, yc_train):.4f} "
        f"test={model.score(Xc_test, yc_test):.4f}"
    )
```

Expected output:

```text
logistic: train=0.9883 test=0.9860
linear SVM: train=0.9906 test=0.9860
RBF SVM: train=0.9789 test=0.9790
LinearSVC: train=0.9906 test=0.9720
```

All four are excellent on this dataset. The nonlinear RBF kernel does not help
here, which is a reminder that a more flexible model is not automatically better.
The linear models already capture the signal.

Inspect the best linear model's errors:

```python
best = models["linear SVM"]
y_pred = best.predict(Xc_test)

print(confusion_matrix(yc_test, y_pred))
print(classification_report(yc_test, y_pred, target_names=cancer.target_names))
```

Expected output:

```text
[[52  1]
 [ 1 89]]
              precision    recall  f1-score   support

   malignant       0.98      0.98      0.98        53
      benign       0.99      0.99      0.99        90

    accuracy                           0.99       143
   macro avg       0.99      0.99      0.99       143
weighted avg       0.99      0.99       0.99       143
```

Only two test cases are misclassified. The labels in scikit-learn are `0 =
malignant` and `1 = benign`.

## 11. Tune `C` and `gamma` with Grid Search

For the RBF kernel the two knobs interact. Let cross-validation find a good pair on
the training set, then score once on the test set.

```python
param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.01, 0.1, 1],
}

search = GridSearchCV(
    make_pipeline(StandardScaler(), SVC(kernel="rbf")),
    param_grid,
    cv=5,
    n_jobs=-1,
)
search.fit(Xc_train, yc_train)

print("best parameters:", search.best_params_)
print("best CV score:", round(search.best_score_, 4))
print("test score:", round(search.score(Xc_test, yc_test), 4))
```

Expected output:

```text
best parameters: {'svc__C': 1, 'svc__gamma': 0.01}
best CV score: 0.9718
test score: 0.9720
```

The best RBF model is slightly worse than the plain linear SVM from the previous
section. That is a legitimate outcome. The grid search did its job: it found the
best configuration in the search space, and that configuration still lost. Report
it honestly rather than forcing the more complex model.

## 12. Try It Yourself

1. **Recover the margin.** For the linear SVM on the blobs, compute `1 / ||w||`
   from `svc.coef_`. How wide is the margin band?
2. **Break `C`.** Set `C=1e6` on the noisy moons data and draw the boundary. Find
   the smallest `C` that still looks smooth.
3. **Try a polynomial kernel.** Set `degree=3` and `coef0=1` on the moons data.
   Does it beat RBF?
4. **Remove the scaler.** Delete `StandardScaler` from the breast cancer pipeline
   and compare test accuracy. Explain the change.
5. **Balance the classes.** Use `class_weight="balanced"` on a dataset with
   imbalanced classes and compare recall before and after.
6. **Use decision scores.** Replace `predict` with `decision_function` and explain
   why the raw scores are useful even without probabilities.

## 13. What We Learned

- The SVM chooses the separating hyperplane with the widest margin.
- The margin width is `2 / ||w||`; the closest points are the support vectors.
- `C` trades margin width against training violations and controls overfitting.
- Kernels replace dot products with similarities, enabling nonlinear boundaries.
- `gamma` controls the reach of the RBF kernel; it must be tuned with `C`.
- Scaling is mandatory, and a pipeline keeps it leak-free.
- On real data, a simpler linear model can match or beat a tuned nonlinear one.
- Always compare against a baseline and report the honest test score.

## Outcome

Learners can compute distances to a hyperplane, identify support vectors, explain
the role of `C`, apply linear and RBF kernels, tune `C` and `gamma` with
cross-validation, demonstrate why scaling matters, and choose an SVM configuration
by comparing it fairly with logistic regression on held-out data.
