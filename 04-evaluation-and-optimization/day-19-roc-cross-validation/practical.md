# Day 19 — Practical: ROC Curves and Cross-Validation

**Module:** Module 5: Evaluation and Optimization

In this practical we will:

- Build an ROC curve and compute AUC.
- Compare three classifiers by their ROC curves.
- Build a precision-recall curve and compute average precision.
- See why ROC can flatter a model on imbalanced data.
- Run k-fold and stratified cross-validation.
- Report mean and spread across folds.
- Compare `KFold` and `StratifiedKFold` on an imbalanced task.
- Demonstrate how preprocessing before CV inflates scores through leakage.

All data comes from scikit-learn.

## 1. Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_validate,
    StratifiedKFold,
    KFold,
    RepeatedStratifiedKFold,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

RANDOM_STATE = 42
```

## 2. ROC Curve for Logistic Regression

Use the same breast-cancer split as Days 17 and 18.

```python
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
logreg.fit(X_train, y_train)

scores = logreg.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, scores)
auc_value = roc_auc_score(y_test, scores)

print("AUC:", round(auc_value, 4))
print("number of ROC points:", len(fpr))
```

Expected output:

```text
AUC: 0.9977
number of ROC points: 6
```

scikit-learn drops intermediate points that lie on straight segments. Six points are
enough to draw this curve because the model separates the classes almost perfectly.

Plot it:

```python
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"logistic regression (AUC = {auc_value:.4f})")
plt.plot([0, 1], [0, 1], "k--", label="random guessing")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("ROC curve")
plt.legend()
plt.show()
```

The curve hugs the top-left corner. The diagonal is the random baseline.

## 3. Reading ROC Points

Each point corresponds to a threshold. Out of the six points, look at the first,
middle, and last.

```python
for i in [1, 3, 4, len(fpr) - 1]:
    print(
        f"fpr={fpr[i]:.4f} tpr={tpr[i]:.4f} "
        f"threshold={thresholds[i]:.4g}"
    )
```

Expected output:

```text
fpr=0.0000 tpr=0.0111 threshold=1
fpr=0.0189 tpr=0.8778 threshold=0.8929
fpr=0.0189 tpr=1.0000 threshold=0.3831
fpr=1.0000 tpr=1.0000 threshold=5.311e-23
```

Reading the curve from right to left (lowering the threshold):

- At threshold ≈ 0.89, TPR is 0.88 with FPR 0.019. The model catches most positives
  with very few false alarms.
- At threshold ≈ 0.38, TPR reaches 1.0 but FPR rises slightly to 0.019 (the same FPR
  because no additional negatives were crossed).
- At threshold ≈ 0, everything is positive: TPR = FPR = 1.

The first point at `tpr=0.0111` corresponds to a single benign example scored near
1.0. Because `pos_label` defaults to 1 (benign), the ROC is computed with benign as
the positive class. This is the same `pos_label` trap from Day 18. If you intend to
treat malignant as positive, pass `pos_label=0`.

## 4. Compare Models by ROC

```python
models = {
    "logistic regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    ),
    "knn (k=5)": make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=5)
    ),
    "majority dummy": DummyClassifier(strategy="most_frequent"),
}

plt.figure(figsize=(6, 6))

for name, model in models.items():
    model.fit(X_train, y_train)
    s = model.predict_proba(X_test)[:, 1]
    f, t, _ = roc_curve(y_test, s)
    a = roc_auc_score(y_test, s)
    plt.plot(f, t, label=f"{name} (AUC = {a:.4f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("ROC comparison")
plt.legend()
plt.show()
```

Expected AUC values:

```text
logistic regression: 0.9977
knn (k=5): 0.9845
majority dummy: 0.5000
```

Logistic regression and KNN are both excellent on this dataset. The dummy is exactly
at random, which is what a constant score should produce.

## 5. Precision-Recall Curve and Average Precision

```python
precision, recall, pr_thresholds = precision_recall_curve(y_test, scores)
ap = average_precision_score(y_test, scores)

print("average precision:", round(ap, 4))
print("baseline (positive rate):", round(y_test.mean(), 4))
```

Expected output:

```text
average precision: 0.9986
baseline (positive rate): 0.6294
```

For breast cancer the positive class is 63% of the test set, so a random classifier
would have average precision around 0.63. The model's 0.9986 is far above that.

```python
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label=f"logistic regression (AP = {ap:.4f})")
plt.axhline(y_test.mean(), color="k", linestyle="--", label="baseline")
plt.xlabel("recall")
plt.ylabel("precision")
plt.title("precision-recall curve")
plt.legend()
plt.show()
```

## 6. ROC versus PR on Imbalanced Data

Rebuild the imbalanced dataset from Day 18 and compare the two summaries.

```python
X_imb, y_imb = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=5,
    n_redundant=5,
    weights=[0.95, 0.05],
    class_sep=2.0,
    flip_y=0.02,
    random_state=RANDOM_STATE,
)

X_train, X_test, y_train, y_test = train_test_split(
    X_imb, y_imb, test_size=0.25, stratify=y_imb, random_state=RANDOM_STATE
)

imb_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
imb_model.fit(X_train, y_train)
imb_scores = imb_model.predict_proba(X_test)[:, 1]

print("positive rate:", round(y_test.mean(), 4))
print("ROC AUC:", round(roc_auc_score(y_test, imb_scores), 4))
print("average precision:", round(average_precision_score(y_test, imb_scores), 4))
```

Expected output:

```text
positive rate: 0.06
ROC AUC: 0.8469
average precision: 0.5811
```

The AUC of 0.85 sounds strong. Average precision is 0.58, and the baseline is only
0.06, so the model is far better than random, but the PR number is more honest about
the difficulty: at many operating points, a large share of alerts are false.

Plot both curves side by side:

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

f, t, _ = roc_curve(y_test, imb_scores)
axes[0].plot(f, t)
axes[0].plot([0, 1], [0, 1], "k--")
axes[0].set_title("ROC, imbalanced data")
axes[0].set_xlabel("false positive rate")
axes[0].set_ylabel("true positive rate")

p, r, _ = precision_recall_curve(y_test, imb_scores)
axes[1].plot(r, p)
axes[1].axhline(y_test.mean(), color="k", linestyle="--")
axes[1].set_title("PR, imbalanced data")
axes[1].set_xlabel("recall")
axes[1].set_ylabel("precision")

plt.tight_layout()
plt.show()
```

## 7. k-Fold Cross-Validation

```python
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print("fold scores:", np.round(scores, 4))
print("mean:", round(scores.mean(), 4))
print("std:", round(scores.std(), 4))
```

Expected output:

```text
fold scores: [0.9825 0.9825 0.9737 0.9737 0.9912]
mean: 0.9807
std: 0.0065
```

The folds vary from 0.9737 to 0.9912. Reporting only one train/test split would hide
that range. The standard deviation of 0.0065 is small, so the model is stable on this
dataset.

## 8. Explicit Stratified Folds

`cross_val_score` already stratifies for a classifier, but making the splitter
explicit gives control over shuffling and reproducibility.

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
print("fold scores:", np.round(scores, 4))
print("mean:", round(scores.mean(), 4))
print("std:", round(scores.std(), 4))
```

Expected output:

```text
fold scores: [0.9737 0.9474 0.9649 0.9912 0.9912]
mean: 0.9737
std: 0.0166
```

Shuffling picks different fold boundaries, so the numbers change. The mean drops
slightly and the spread widens. This is normal sampling variation, not a different
model.

## 9. Multiple Metrics at Once

`cross_validate` returns several scores per fold.

```python
results = cross_validate(
    model,
    X,
    y,
    cv=skf,
    scoring=["accuracy", "precision", "recall", "roc_auc"],
)

for key in ["test_accuracy", "test_precision", "test_recall", "test_roc_auc"]:
    values = results[key]
    print(
        f"{key:16s} fold={np.round(values, 4)} "
        f"mean={values.mean():.4f}"
    )
```

Expected output:

```text
test_accuracy    fold=[0.9737 0.9474 0.9649 0.9912 0.9912] mean=0.9737
test_precision   fold=[0.9857 0.9221 0.9474 1.     0.9861] mean=0.9683
test_recall     fold=[0.9718 1.     1.     0.9861 1.    ] mean=0.9916
test_roc_auc     fold=[0.9951 0.9977 0.9851 1.     0.9987] mean=0.9953
```

The model's recall is higher than its precision: it misses very few benign cases but
produces a few false alarms. AUC is above 0.985 in every fold. Reporting all four
metrics gives a complete picture at almost no extra cost.

## 10. KFold versus StratifiedKFold on Imbalanced Data

Build a small imbalanced dataset with about 10% positives.

```python
X_imb, y_imb = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=5,
    weights=[0.9, 0.1],
    class_sep=1.5,
    random_state=RANDOM_STATE,
)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

for name, cv in [
    ("KFold", KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)),
    ("StratifiedKFold", StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE)),
]:
    scores = cross_val_score(model, X_imb, y_imb, cv=cv, scoring="recall")
    print(
        f"{name:16s} folds={np.round(scores, 4)} "
        f"mean={scores.mean():.4f} std={scores.std():.4f}"
    )
```

Expected output:

```text
KFold            folds=[0.6364 0.625  0.4286 1.     0.8667] mean=0.7113 std=0.2002
StratifiedKFold  folds=[0.8    0.7    0.8    0.7273 0.7273] mean=0.7509 std=0.0413
```

Plain KFold produces folds ranging from 0.43 to 1.00, a standard deviation of 0.20,
because the number of positives per fold fluctuates. StratifiedKFold keeps the ratio
close to 10% in every fold, cutting the standard deviation to 0.04. The mean is also
slightly higher and much more trustworthy.

## 11. Repeated Cross-Validation

One 5-fold run gives five numbers. Repeating with different shuffles gives a more
stable estimate.

```python
repeated = RepeatedStratifiedKFold(
    n_splits=5, n_repeats=3, random_state=RANDOM_STATE
)

scores = cross_val_score(model, X_imb, y_imb, cv=repeated, scoring="recall")
print("number of scores:", len(scores))
print("mean:", round(scores.mean(), 4))
print("std:", round(scores.std(), 4))
```

Expected output:

```text
number of scores: 15
mean: 0.7642
std: 0.1187
```

Fifteen estimates instead of five. The mean is close to the stratified single run, and
the spread reflects both fold-to-fold and repeat-to-repeat variation.

## 12. Leakage: Preprocessing Before Cross-Validation

This example uses 200 samples with 2,000 features, only 20 of which carry a weak
signal. The wrong pipeline selects features on all rows before CV.

```python
rng = np.random.default_rng(0)
X_noise = rng.normal(size=(200, 2000))
y_noise = rng.integers(0, 2, 200)
X_noise[y_noise == 1, :20] += 0.35

# WRONG: feature selection sees every fold's test rows
selector = SelectKBest(f_classif, k=20).fit(X_noise, y_noise)
X_selected = selector.transform(X_noise)
wrong = cross_val_score(
    LogisticRegression(max_iter=1000), X_selected, y_noise, cv=5
).mean()

# RIGHT: selection is fit inside each fold
right = cross_val_score(
    make_pipeline(
        SelectKBest(f_classif, k=20),
        LogisticRegression(max_iter=1000),
    ),
    X_noise,
    y_noise,
    cv=5,
).mean()

print("wrong (selection outside CV):", round(wrong, 4))
print("right (selection inside CV):", round(right, 4))
```

Expected output:

```text
wrong (selection outside CV): 0.78
right (selection inside CV): 0.495
```

The wrong approach reports 0.78; the honest estimate is 0.495, essentially chance.
The selector had inspected the test rows and chosen features that looked predictive
across the whole dataset, including the holdout. This is why the pipeline pattern is
mandatory, not optional.

## 13. Try It Yourself

1. **Use `pos_label=0`.** Recompute the ROC and AUC with malignant as positive.
   Does AUC change?
2. **Compare scoring rules.** Plot precision-recall curves for logistic regression
   and KNN on the imbalanced data. Which has higher average precision?
3. **Vary k.** Run 3, 5, 10, and 20-fold CV on breast cancer. How do the mean and
   standard deviation change?
4. **Remove stratification.** Re-run the imbalanced experiment with `StratifiedKFold`
   replaced by `KFold` and no shuffle. Explain the result.
5. **Group the data.** Create a `group` column with 50 groups, then use `GroupKFold`.
   Why would plain k-fold be optimistic?
6. **Nested CV.** Wrap `GridSearchCV` inside `cross_val_score` and compare the
   resulting estimate with the inner best score.
7. **Time series.** Use `TimeSeriesSplit` on a shuffled-out time series and explain
   why shuffling is wrong.

## 14. What We Learned

- ROC plots TPR against FPR across thresholds; AUC summarizes ranking quality.
- A perfect model hugs the top-left; random guessing is the diagonal.
- `pos_label` determines which class is treated as positive, for ROC as well as for
  precision and recall.
- The PR curve and average precision focus on the positive class.
- Under strong imbalance, AUC can look strong while average precision reveals
  difficulty.
- Cross-validation gives a mean and a spread across folds.
- Stratified folds reduce variance when classes are imbalanced.
- Repeated CV gives a more stable estimate than a single run.
- Preprocessing outside CV leaks test information and inflates scores.
- Pipelines keep every learned transformation inside each fold.

## Outcome

Learners can build and interpret ROC and precision-recall curves, compute AUC and
average precision, choose the appropriate curve for imbalanced data, run k-fold and
stratified cross-validation, report mean and spread, explain repeated CV, and avoid
leakage by placing all preprocessing inside a pipeline.
