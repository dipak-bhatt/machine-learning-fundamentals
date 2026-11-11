# Day 18 — Practical: Classification Metrics

**Module:** Module 5: Evaluation and Optimization

In this practical we will:

- Compute a confusion matrix on a real dataset.
- Read precision, recall, and F1 from it.
- Show a trivial model beating a real one on accuracy.
- Compare default and class-weighted logistic regression on imbalanced data.
- Sweep the decision threshold and watch precision and recall trade off.
- Choose a threshold from a cost matrix.
- Evaluate a multi-class classifier with macro, weighted, and micro averages.

All data is built into scikit-learn.

## 1. Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
)

RANDOM_STATE = 42
```

## 2. Confusion Matrix on Breast Cancer

We reuse the split from Day 17. The positive class here is malignant (`target = 0`),
which is the class we most want to catch.

```python
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print("accuracy:", round(accuracy_score(y_test, y_pred), 4))
```

Expected output:

```text
[[52  1]
 [ 1 89]]
accuracy: 0.986
```

Read the matrix by rows: actual class, then predicted class.

- Row 0 (malignant, 53 cases): 52 predicted malignant, 1 predicted benign.
- Row 1 (benign, 90 cases): 1 predicted malignant, 89 predicted benign.

So there are 2 errors in 143 cases: one false negative (a missed cancer) and one
false positive (an unnecessary alarm).

```python
ConfusionMatrixDisplay(
    cm, display_labels=cancer.target_names
).plot(cmap="Blues")
plt.title("confusion matrix, logistic regression")
plt.show()
```

## 3. Precision, Recall, and F1

```python
print("precision:", round(precision_score(y_test, y_pred), 4))
print("recall:", round(recall_score(y_test, y_pred), 4))
print("f1:", round(f1_score(y_test, y_pred), 4))
```

Expected output:

```text
precision: 0.9889
recall: 0.9889
f1: 0.9889
```

Check the arithmetic from the matrix. The positive class is malignant:

```text
TP = 52, FN = 1, FP = 1, TN = 89
precision = 52 / (52 + 1) = 0.9811
recall    = 52 / (52 + 1) = 0.9811
```

The aggregate metrics reported above are 0.9889 because scikit-learn's
`pos_label` defaults to 1, which is *benign* here. This is an important practical
trap: always confirm which class is positive.

```python
for pos in [0, 1]:
    print(
        f"pos_label={pos} ({cancer.target_names[pos]}): "
        f"precision={precision_score(y_test, y_pred, pos_label=pos):.4f} "
        f"recall={recall_score(y_test, y_pred, pos_label=pos):.4f} "
        f"f1={f1_score(y_test, y_pred, pos_label=pos):.4f}"
    )
```

Expected output:

```text
pos_label=0 (malignant): precision=0.9811 recall=0.9811 f1=0.9811
pos_label=1 (benign): precision=0.9889 recall=0.9889 f1=0.9889
```

`classification_report` prints both classes and the averages:

```python
print(classification_report(
    y_test, y_pred, target_names=cancer.target_names, digits=4
))
```

Expected output:

```text
              precision    recall  f1-score   support

   malignant     0.9811    0.9811    0.9811        53
      benign     0.9889    0.9889    0.9889        90

    accuracy                         0.9860       143
   macro avg     0.9850    0.9850    0.9850       143
weighted avg     0.9860    0.9860    0.9860       143
```

## 4. The Accuracy Paradox

Now build an imbalanced dataset: 95% negative, 5% positive. Compare a model that
always predicts the majority class with logistic regression.

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

print("positive rate in test:", round(y_test.mean(), 4))
```

Expected output:

```text
positive rate in test: 0.06
```

The test set contains 30 positives out of 500 rows: 6%, strongly imbalanced.

```python
dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)

print("dummy accuracy:", round(accuracy_score(y_test, pred_dummy), 4))
print("dummy recall:", round(recall_score(y_test, pred_dummy), 4))
```

Expected output:

```text
dummy accuracy: 0.94
dummy recall: 0.0
```

The dummy is 94% accurate and catches nothing. This is the accuracy paradox. Accuracy
alone would rank it highly.

Logistic regression at the default threshold:

```python
logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
logreg.fit(X_train, y_train)
pred_lr = logreg.predict(X_test)

print("accuracy:", round(accuracy_score(y_test, pred_lr), 4))
print("precision:", round(precision_score(y_test, pred_lr, zero_division=0), 4))
print("recall:", round(recall_score(y_test, pred_lr), 4))
print("f1:", round(f1_score(y_test, pred_lr), 4))
print(confusion_matrix(y_test, pred_lr))
```

Expected output:

```text
accuracy: 0.96
precision: 0.7778
recall: 0.4667
f1: 0.5833
[[466   4]
 [ 16  14]]
```

The model is 96% accurate, better than the dummy, yet it misses 16 of 30 positives.
Precision looks strong (0.78) because it makes few positive predictions. Recall tells
the rest of the story: it finds less than half the cases.

## 5. Class Weights

Tell the model that positive errors cost more.

```python
balanced = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)
balanced.fit(X_train, y_train)
pred_bal = balanced.predict(X_test)

print("accuracy:", round(accuracy_score(y_test, pred_bal), 4))
print("precision:", round(precision_score(y_test, pred_bal, zero_division=0), 4))
print("recall:", round(recall_score(y_test, pred_bal), 4))
print("f1:", round(f1_score(y_test, pred_bal), 4))
print(confusion_matrix(y_test, pred_bal))
```

Expected output:

```text
accuracy: 0.856
precision: 0.2667
recall: 0.8
f1: 0.4
[[404  66]
 [  6  24]]
```

Recall jumps from 0.47 to 0.80: the model now catches 24 of 30 positives. Accuracy
falls from 0.96 to 0.856 because false positives rose from 4 to 66. Precision falls
from 0.78 to 0.27. Whether that trade is good depends entirely on the cost of a
missed positive versus a false alarm.

## 6. Threshold Sweep

Class weights are one way to move the operating point. Another is to change the
threshold on the predicted probability.

```python
proba = logreg.predict_proba(X_test)[:, 1]

rows = []
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    pred = (proba >= threshold).astype(int)
    rows.append({
        "threshold": threshold,
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    })

print(pd.DataFrame(rows).round(4).to_string(index=False))
```

Expected output:

```text
 threshold  precision  recall      f1
       0.1     0.3692  0.8000  0.5053
       0.2     0.4524  0.6333  0.5278
       0.3     0.6000  0.6000  0.6000
       0.4     0.7727  0.5667  0.6538
       0.5     0.7778  0.4667  0.5833
       0.6     0.7692  0.3333  0.4651
       0.7     0.6667  0.1333  0.2222
```

The trade-off is clear:

- At 0.1, recall is 0.80 but precision is 0.37.
- At 0.5, precision is 0.78 but recall is 0.47.
- F1 peaks at 0.4, not at the default 0.5.

Plot the curves to see the shape:

```python
thresholds = np.linspace(0.05, 0.95, 91)
precisions, recalls = [], []

for t in thresholds:
    pred = (proba >= t).astype(int)
    precisions.append(precision_score(y_test, pred, zero_division=0))
    recalls.append(recall_score(y_test, pred))

plt.plot(thresholds, precisions, label="precision")
plt.plot(thresholds, recalls, label="recall")
plt.xlabel("threshold")
plt.ylabel("score")
plt.title("precision and recall versus threshold")
plt.legend()
plt.show()
```

## 7. Choosing a Threshold from Costs

Suppose a missed positive costs 10 units and a false alarm costs 1. Compute the total
cost at each threshold and pick the cheapest.

```python
best = None

for t in np.arange(0.05, 0.96, 0.05):
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    cost = 10 * fn + 1 * fp

    if best is None or cost < best[1]:
        best = (round(t, 2), cost, tp, fp, fn)

print("best threshold:", best[0])
print("cost:", best[1])
print("tp, fp, fn:", best[2], best[3], best[4])
```

Expected output:

```text
best threshold: 0.15
cost: 98
tp, fp, fn: 23 28 7
```

The cost-minimizing threshold is 0.15, far from 0.5. At 0.5 the cost would be 164.
The right threshold depends on the 10:1 cost ratio, not on a convention.

This is what it means to make the metric follow the decision. If the ratio changes,
recompute. If the costs are unknown, report the curve and let the stakeholder
choose.

## 8. Multi-Class Metrics on Iris

Iris has three classes. The confusion matrix is 3x3.

```python
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, stratify=iris.target,
    random_state=RANDOM_STATE
)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print("accuracy:", round(accuracy_score(y_test, y_pred), 4))
```

Expected output:

```text
[[15  0  0]
 [ 0 14  1]
 [ 0  3 12]]
accuracy: 0.9111
```

Fifteen setosa are perfect. One versicolor is called virginica, and three virginica
are called versicolor. Those two species overlap in feature space.

```python
print(classification_report(
    y_test, y_pred, target_names=iris.target_names, digits=4
))
```

Expected output:

```text
              precision    recall  f1-score   support

      setosa     1.0000    1.0000    1.0000        15
  versicolor     0.8235    0.9333    0.8750        15
   virginica     0.9231    0.8000    0.8571        15

    accuracy                         0.9111        45
   macro avg     0.9155    0.9111    0.9107        45
weighted avg     0.9155    0.9111    0.9107        45
```

Now compare the three averages explicitly:

```python
for average in ["macro", "weighted", "micro"]:
    p, r, f, _ = precision_recall_fscore_support(
        y_test, y_pred, average=average
    )
    print(f"{average}: precision={p:.4f} recall={r:.4f} f1={f:.4f}")
```

Expected output:

```text
macro: precision=0.9155 recall=0.9111 f1=0.9107
weighted: precision=0.9155 recall=0.9111 f1=0.9107
micro: precision=0.9111 recall=0.9111 f1=0.9111
```

Macro and weighted agree here because every class has exactly 15 test examples, so
weighting by support changes nothing. Micro equals accuracy, as it must in
single-label multi-class classification.

## 9. Try It Yourself

1. **Break the balance.** Change the iris split to stratify on a binarized target
   with 90% one class. Compare macro and weighted F1.
2. **Use F-beta.** On the imbalanced dataset, compute `fbeta_score` with
   `beta=2` and `beta=0.5`. Which threshold wins for each?
3. **Try SMOTE-like resampling.** Oversample the minority class with
   `sklearn.utils.resample` and compare recall before and after.
4. **Change the cost ratio.** Rerun the cost section with `c_FN/c_FP = 2` and then
   `50`. How does the best threshold move?
5. **Add a baseline.** Compare logistic regression with `DummyClassifier` on
   precision and recall, not just accuracy.
6. **Check calibration.** Bin predicted probabilities and plot observed positive
   rates. Does a score of 0.8 mean an 80% chance?

## 10. What We Learned

- The confusion matrix reveals the kinds of errors that accuracy hides.
- Precision and recall measure different failure modes and must be read together.
- `pos_label` matters: the same predictions give different precision and recall
  depending on which class is treated as positive.
- A 94%-accurate dummy classifier can be worthless.
- Class weights raise recall at the cost of precision.
- The threshold is a decision variable; 0.5 is not sacred.
- The cost-minimizing threshold follows the ratio of error costs.
- Multi-class metrics need an explicit averaging choice; macro and weighted differ
  when classes are unbalanced.
- Micro F1 equals accuracy for single-label multi-class problems.

## Outcome

Learners can compute and interpret a confusion matrix, precision, recall,
specificity, F1, and balanced accuracy, diagnose the accuracy paradox, apply class
weights and threshold tuning, choose a threshold from a cost matrix, and report
multi-class results with the appropriate averaging.
