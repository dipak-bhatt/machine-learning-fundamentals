# Day 07 — Practical: Logistic Regression and Classification

**Module:** Module 2: Supervised Learning

In this practical we build our first classifier. We will:

- Compute the sigmoid by hand and perform one logistic regression update.
- Load the Breast Cancer Wisconsin dataset from scikit-learn.
- Compare a classifier against a "predict the majority class" baseline.
- Train logistic regression and inspect predicted probabilities.
- Read a confusion matrix and compute precision, recall, and F1.
- Move the classification threshold and watch the tradeoff change.
- Interpret coefficients as odds ratios.
- Train a multiclass classifier on the Iris data.

The notebook should be created inside this Day 07 folder. Both datasets are built
into scikit-learn, so there is no CSV path to configure and no download step.

## 1. Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    log_loss,
    roc_auc_score,
    classification_report,
)

RANDOM_STATE = 42
```

`RANDOM_STATE` keeps the split repeatable. The `make_pipeline` helper will bundle
scaling and the model so they are fitted together on the training rows only.

## 2. Warm-Up: The Sigmoid and One Update

The sigmoid turns any real number into a probability between 0 and 1:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


for z in [-2, 0, 2]:
    print(z, round(float(sigmoid(z)), 3))
```

Expected output:

```text
-2 0.119
0 0.5
2 0.881
```

At `z = 0` the model is perfectly undecided. Negative scores push the probability
toward 0, and positive scores push it toward 1. For very large negative `z`, the
expression `np.exp(-z)` can overflow; scikit-learn uses a numerically stable version
internally. The direct formula is fine for the moderate values used here.

Now perform one logistic regression update. We have one row with `x = 1` and true
label `y = 1`. Start with `b = 0`, `w = 0`, and use a deliberately large learning
rate `alpha = 1.0` so the change is visible.

```python
x, y = 1.0, 1.0
b, w, alpha = 0.0, 0.0, 1.0

z = b + w * x
p = sigmoid(z)
gradient_b = p - y
gradient_w = (p - y) * x

b = b - alpha * gradient_b
w = w - alpha * gradient_w

print("z:", z)
print("probability:", p)
print("gradients:", gradient_b, gradient_w)
print("new b, w:", b, w)
print("new probability:", round(float(sigmoid(b + w * x)), 3))
```

Expected output:

```text
z: 0.0
probability: 0.5
gradients: -0.5 -0.5
new b, w: 0.5 0.5
new probability: 0.731
```

The probability moved from 0.5 to 0.731, toward the correct label. The gradient is
`p - y`, exactly as derived in the theory file. One row is not enough to learn a
useful classifier; the real model sees hundreds of rows.

## 3. Load and Inspect the Breast Cancer Data

The Breast Cancer Wisconsin dataset contains measurements from digitized images of
cell nuclei. Each row is a sample, and the target says whether the sample is
malignant or benign.

```python
cancer = load_breast_cancer(as_frame=True)
df = cancer.frame

print("Shape:", df.shape)
print(df["target"].value_counts())
print("Feature names (first 5):", [str(name) for name in cancer.feature_names[:5]])
print("Target names:", [str(name) for name in cancer.target_names])
```

Expected output:

```text
Shape: (569, 31)
target
1    357
0    212
Name: count, dtype: int64
Feature names (first 5): ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
Target names: ['malignant', 'benign']
```

There are 569 rows and 30 features plus the target. In scikit-learn's encoding,
`target = 0` means **malignant** and `target = 1` means **benign**. That is the
opposite of the convention we want for screening, where malignant should be the
positive class. We therefore create our own target column:

```python
X = df[cancer.feature_names].copy()
y = (df["target"] == 0).astype(int)

print("Positive (malignant) rows:", int(y.sum()))
print("Negative (benign) rows:", int((1 - y).sum()))
```

Expected output:

```text
Positive (malignant) rows: 212
Negative (benign) rows: 357
```

Now positive (`1`) means malignant. Keep that in mind for every metric: recall and
precision will refer to finding malignant tumors.

Look at a few feature ranges. The features are measured on very different scales,
which is why we will standardize them before fitting.

```python
df[["mean radius", "mean texture", "mean perimeter", "mean area"]].describe().round(2)
```

Expected output:

```text
       mean radius  mean texture  mean perimeter  mean area
count       569.00        569.00          569.00     569.00
mean         14.13         19.29           91.97     654.89
std           3.52          4.30           24.30     351.91
min           6.98          9.71           43.79     143.50
25%          11.70         16.17           75.17     420.30
50%          13.37         18.84           86.24     551.10
75%          15.78         21.80          104.10     782.70
max          28.11         39.28          188.50    2501.00
```

`mean area` ranges from about 144 to 2,501, while `mean texture` stays between about
10 and 39. Standardization puts them on a comparable footing.

## 4. Split the Data Before Doing Anything Else

We split once and reuse the same test rows for every model. `stratify=y` keeps the
proportion of malignant and benign rows the same in both parts.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Malignant in test:", int(y_test.sum()))
print("Benign in test:", int((1 - y_test).sum()))
```

Expected output:

```text
Training rows: 455
Test rows: 114
Malignant in test: 42
Benign in test: 72
```

The split happens **before** any scaling. The scaler will learn its mean and
standard deviation from `X_train` only, so no test information leaks into training.

## 5. Establish a Baseline

The simplest classifier predicts the majority class every time. Here the majority
is benign, with 72 of 114 test rows.

```python
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

print("Baseline accuracy:", round(accuracy_score(y_test, baseline_pred), 3))
print("Baseline recall (malignant):", round(recall_score(y_test, baseline_pred), 3))
print(confusion_matrix(y_test, baseline_pred))
```

Expected output:

```text
Baseline accuracy: 0.632
Baseline recall (malignant): 0.0
[[72  0]
 [42  0]]
```

The baseline looks 63% accurate, but it never predicts malignant. Its recall is
zero, which means it misses every cancer. This is the accuracy trap in one line:
**always report more than accuracy**.

## 6. Train Logistic Regression

We standardize the features and fit logistic regression. The pipeline fits the
scaler on the training rows and applies the same transformation to the test rows.

```python
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
).fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
test_proba = model.predict_proba(X_test)[:, 1]

print("Train accuracy:", round(accuracy_score(y_train, train_pred), 3))
print("Test accuracy:", round(accuracy_score(y_test, test_pred), 3))
print("Test log loss:", round(log_loss(y_test, test_proba), 3))
print("Test ROC AUC:", round(roc_auc_score(y_test, test_proba), 3))
```

Expected output:

```text
Train accuracy: 0.987
Test accuracy: 0.965
Test log loss: 0.077
Test ROC AUC: 0.996
```

The model is accurate on both parts. The small gap between training and test
accuracy is normal and not alarming here. The log loss of 0.077 is much better than
the baseline we compute in Section 10, and the ROC AUC of 0.996 says the model ranks
malignant samples above benign ones almost perfectly across all thresholds.

`predict` applies the default threshold of 0.5. `predict_proba` gives the
probability itself, which is more informative.

## 7. Inspect the Predicted Probabilities

A classifier's probability is often more useful than its hard label. Here are the
first five test rows:

```python
results = pd.DataFrame({
    "actual_malignant": y_test.values,
    "prob_malignant": test_proba,
    "predicted_malignant": test_pred,
})
results.head().round(4)
```

Expected output:

```text
   actual_malignant  prob_malignant  predicted_malignant
0                 0          0.0004                    0
1                 1          1.0000                    1
2                 0          0.0426                    0
3                 1          0.5762                    1
4                 0          0.5092                    1
```

Rows 0, 1, and 2 are easy. Row 3 is malignant with a probability of 0.576, so it is
barely above the threshold. Row 4 is actually benign but receives a probability of
0.509, so it becomes a false positive. These borderline cases are exactly where the
threshold matters, as the next section shows.

Seeing a probability also lets a human decide what to do. A 0.51 and a 0.99 are not
the same level of concern, even though both become "malignant" at the default
threshold.

## 8. Read the Confusion Matrix

The confusion matrix counts true and predicted labels. Rows are actual labels and
columns are predicted labels.

```python
cm = confusion_matrix(y_test, test_pred)
print(cm)
print(classification_report(y_test, test_pred, target_names=["benign", "malignant"], digits=3))
```

Expected output:

```text
[[71  1]
 [ 3 39]]
              precision    recall  f1-score   support

      benign      0.959     0.986     0.973        72
   malignant      0.975     0.929     0.951        42

    accuracy                          0.965       114
   macro avg      0.967     0.957     0.962       114
weighted avg      0.965     0.965     0.965       114
```

Reading the matrix:

- 71 benign rows were correctly cleared (true negatives).
- 1 benign row was flagged as malignant (false positive).
- 3 malignant rows were missed (false negatives).
- 39 malignant rows were correctly found (true positives).

The malignant row is the important one: precision 0.975, recall 0.929. The model
finds about 93% of cancers and, when it raises an alarm, is right about 97.5% of the
time. The three missed cases are the number a screening program would try to reduce.

A visual version of the same counts:

```python
ConfusionMatrixDisplay(cm, display_labels=["benign", "malignant"]).plot(cmap="Blues")
plt.title("Confusion matrix at threshold 0.5")
plt.show()
```

The colors make it obvious that most predictions fall on the diagonal, with a small
number of errors off it.

## 9. Change the Threshold

The model's probabilities do not change when we move the threshold; only the labels
do. Compare three thresholds:

```python
for threshold in [0.3, 0.5, 0.7]:
    pred_at = (test_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_at).ravel()
    print(
        f"threshold {threshold}: TP={tp} FP={fp} FN={fn} TN={tn} "
        f"accuracy={accuracy_score(y_test, pred_at):.3f} "
        f"precision={precision_score(y_test, pred_at):.3f} "
        f"recall={recall_score(y_test, pred_at):.3f} "
        f"f1={f1_score(y_test, pred_at):.3f}"
    )
```

Expected output:

```text
threshold 0.3: TP=41 FP=1 FN=1 TN=71 accuracy=0.982 precision=0.976 recall=0.976 f1=0.976
threshold 0.5: TP=39 FP=1 FN=3 TN=71 accuracy=0.965 precision=0.975 recall=0.929 f1=0.951
threshold 0.7: TP=38 FP=0 FN=4 TN=72 accuracy=0.965 precision=1.000 recall=0.905 f1=0.950
```

What changed:

- Lowering the threshold to 0.3 catches two more malignant rows (39 to 41 true
  positives) and reduces the misses from three to one. It adds no new false positive
  on this split.
- Raising the threshold to 0.7 removes the false positive, but misses one more
  malignant row. Precision becomes perfect, at the cost of recall.

This is the central tradeoff of classification. A lower threshold is a more cautious
screening policy; a higher threshold is a more conservative alarm. Which is better
depends on the cost of a missed cancer versus the cost of a false alarm.

> Do not pick `0.3` just because it looks best in this table. That would be tuning
> on the test set. In a real project, split the training data into training and
> validation parts, choose the threshold there, then use the test set once for the
> final check.

## 10. Log Loss and the Baseline

Log loss measures the quality of the probabilities, not just the labels. Lower is
better.

```python
prior = DummyClassifier(strategy="prior").fit(X_train, y_train)
prior_proba = prior.predict_proba(X_test)[:, 1]

print("Prior log loss:", round(log_loss(y_test, prior_proba), 3))
print("Model log loss:", round(log_loss(y_test, test_proba), 3))
```

Expected output:

```text
Prior log loss: 0.658
Model log loss: 0.077
```

The prior baseline predicts the same training class frequency for every row. It is
much worse than the model, even though its accuracy was not terrible. Log loss
reveals that its probabilities carry little information.

To see how hard log loss punishes confident mistakes:

```python
for prob, actual in [(0.9, 1), (0.6, 1), (0.999, 0)]:
    loss = -np.log(prob) if actual == 1 else -np.log(1 - prob)
    print(f"p={prob}, actual={actual}, loss={loss:.3f}")
```

Expected output:

```text
p=0.9, actual=1, loss=0.105
p=0.6, actual=1, loss=0.511
p=0.999, actual=0, loss=6.908
```

A confident wrong answer is expensive. That is the behavior we want from a
probability model: it should not be rewarded for being confidently incorrect.

## 11. Interpret the Coefficients

Logistic regression is linear in the log-odds. Since we standardized the features,
each coefficient describes the change in log-odds for a one standard-deviation
increase in that feature.

```python
logistic = model.named_steps["logisticregression"]
coefs = pd.Series(logistic.coef_[0], index=cancer.feature_names)

print("Intercept:", round(float(logistic.intercept_[0]), 3))
print(coefs.sort_values(ascending=False).head(6).round(3).to_string())
print(coefs.sort_values().head(6).round(3).to_string())
```

Expected output:

```text
Intercept: -0.243
worst texture          1.434
radius error           1.233
worst symmetry         1.061
mean concave points    0.953
worst concavity        0.911
area error             0.909
compactness error         -0.907
fractal dimension error   -0.594
mean compactness          -0.440
texture error             -0.408
symmetry error            -0.331
worst compactness         -0.173
```

Positive coefficients increase the odds of the malignant class; negative
coefficients decrease them. The largest positive coefficient belongs to
`worst texture`, followed by `radius error`. The largest negative coefficient
belongs to `compactness error`.

Do not over-read these numbers on their own:

- The features are highly correlated. `mean radius`, `mean perimeter`, and
  `mean area` describe almost the same geometry, so a coefficient can shrink or flip
  when another correlated feature is present.
- These coefficients are on **standardized** features. They are not "per millimeter"
  or "per unit" effects in the original scale.
- A coefficient is an association in this dataset, not proof that changing the
  feature changes the diagnosis.

### Convert one coefficient to an odds ratio

An odds ratio makes the direction and size easier to read. For a coefficient `w`,
a one-unit increase in the standardized feature multiplies the odds by `exp(w)`.
An odds ratio above 1 increases the odds of the malignant class; a value below 1
decreases them.

Now fit a model with only `mean radius`, so the coefficient has a clear meaning
before we standardize it back to the original units.

```python
X_radius = df[["mean radius"]]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_radius, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

radius_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
).fit(Xr_train, yr_train)

radius_lr = radius_model.named_steps["logisticregression"]
radius_scaler = radius_model.named_steps["standardscaler"]
coef_scaled = radius_lr.coef_[0][0]
raw_slope = coef_scaled / radius_scaler.scale_[0]
raw_intercept = radius_lr.intercept_[0] - raw_slope * radius_scaler.mean_[0]

print("Accuracy:", round(accuracy_score(yr_test, radius_model.predict(Xr_test)), 3))
print("Coefficient on standardized radius:", round(float(coef_scaled), 3))
print("Odds ratio per standard deviation:", round(float(np.exp(coef_scaled)), 1))
print("Raw log-odds:", round(float(raw_intercept), 3), "+", round(float(raw_slope), 3), "* radius")
print("Odds ratio per unit:", round(float(np.exp(raw_slope)), 2))
```

Expected output:

```text
Accuracy: 0.86
Coefficient on standardized radius: 3.265
Odds ratio per standard deviation: 26.2
Raw log-odds: -13.566 + 0.913 * radius
Odds ratio per unit: 2.49
```

Radius alone already reaches 86% accuracy. The raw-scale equation is:

```text
log-odds(malignant) = -13.566 + 0.913 * mean_radius
```

A one-unit increase in `mean radius` multiplies the odds of malignancy by about
2.49. A one standard-deviation increase multiplies them by about 26.2. The
probability for a few radii:

```python
for radius in [10, 15, 20]:
    z = raw_intercept + raw_slope * radius
    print(f"radius {radius}: probability malignant = {sigmoid(z):.3f}")
```

Expected output:

```text
radius 10: probability malignant = 0.012
radius 15: probability malignant = 0.533
radius 20: probability malignant = 0.991
```

The probability rises quickly around a radius of 15. That is a property of the
sigmoid, not a universal medical rule. These numbers come from this model and this
dataset.

## 12. Multiclass Classification with Iris

Logistic regression also handles more than two classes. The Iris dataset has three
species and four features.

```python
iris = load_iris(as_frame=True)
Xi = iris.frame[iris.feature_names]
yi = iris.target

Xi_train, Xi_test, yi_train, yi_test = train_test_split(
    Xi, yi, test_size=0.2, random_state=RANDOM_STATE, stratify=yi
)

multiclass_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
).fit(Xi_train, yi_train)

multi_pred = multiclass_model.predict(Xi_test)
multi_proba = multiclass_model.predict_proba(Xi_test)

print("Test accuracy:", round(accuracy_score(yi_test, multi_pred), 3))
print(confusion_matrix(yi_test, multi_pred))
print(classification_report(yi_test, multi_pred, target_names=iris.target_names, digits=3))
```

Expected output:

```text
Test accuracy: 0.933
[[10  0  0]
 [ 0  9  1]
 [ 0  1  9]]
              precision    recall  f1-score   support

      setosa      1.000     1.000     1.000        10
  versicolor      0.900     0.900     0.900        10
   virginica      0.900     0.900     0.900        10

    accuracy                          0.933       30
   macro avg      0.933     0.933     0.933        30
weighted avg      0.933     0.933     0.933        30
```

Setosa is perfectly separated. Versicolor and virginica confuse one sample each,
which is common because those two species overlap in these measurements.

The probabilities form a distribution over the three classes:

```python
print(np.round(multi_proba[:3], 3))
print("row sums:", np.round(multi_proba[:3].sum(axis=1), 3))
print("classes:", multiclass_model.named_steps["logisticregression"].classes_)
```

Expected output:

```text
[[0.979 0.021 0.   ]
 [0.004 0.369 0.627]
 [0.149 0.842 0.009]]
row sums: [1. 1. 1.]
classes: [0 1 2]
```

Each row sums to 1. `predict` simply chooses the class with the largest probability.
For the second row, the model is 62.7% sure of class 2 and 36.9% sure of class 1, so
it predicts class 2.

scikit-learn uses a multinomial model by default here. Compare it with one-vs-rest,
which trains a separate binary classifier per class:

```python
ovr_model = make_pipeline(
    StandardScaler(),
    OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
).fit(Xi_train, yi_train)

print("One-vs-rest accuracy:", round(accuracy_score(yi_test, ovr_model.predict(Xi_test)), 3))
print(confusion_matrix(yi_test, ovr_model.predict(Xi_test)))
```

Expected output:

```text
One-vs-rest accuracy: 0.9
[[10  0  0]
 [ 0  8  2]
 [ 0  1  9]]
```

Multinomial wins on this split, 0.933 to 0.900. That is not a universal rule; it is a
reminder that the multiclass strategy is a choice worth testing. With macro
averages, every class counts equally, which is why the per-class table is more
informative than a single accuracy number.

## 13. Try It Yourself

Use the variables from this notebook.

1. Try thresholds `0.2`, `0.4`, and `0.8`. For each, write down the confusion
   matrix, precision, and recall. Which threshold would you choose for a screening
   tool, and why?
2. Fit the model with `class_weight="balanced"` and compare recall and precision
   with the default model. What changes, and what stays the same?
3. Vary `C` from `0.01` to `100`. Plot or print test accuracy and log loss. What
   happens to the coefficients as the penalty gets stronger?
4. Add `PolynomialFeatures(degree=2)` before the scaler. Does the polynomial model
   improve test log loss, or does it overfit? Compare with the plain model on the
   same split.
5. Use only `mean radius` and `mean texture`, fit a model, and plot its probability
   surface. Where does the 0.5 boundary sit?
6. Drop correlated features such as `mean perimeter` and `mean area`, keeping only
   `mean radius`. How do the coefficients and test metrics change?
7. Compute odds ratios for the top five positive and top five negative coefficients
   in the full model. Explain two of them in plain words.
8. Compare multinomial and one-vs-rest on Iris with macro F1 instead of accuracy.
   Do the two strategies still differ by the same amount?

Hint for question 5:

```python
X_two = df[["mean radius", "mean texture"]]
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X_two, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

two_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
).fit(X2_train, y2_train)

xx, yy = np.meshgrid(
    np.linspace(df["mean radius"].min(), df["mean radius"].max(), 200),
    np.linspace(df["mean texture"].min(), df["mean texture"].max(), 200),
)
grid = pd.DataFrame({"mean radius": xx.ravel(), "mean texture": yy.ravel()})
zz = two_model.predict_proba(grid)[:, 1].reshape(xx.shape)

plt.figure(figsize=(8, 5))
plt.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.7)
plt.colorbar(label="P(malignant)")
plt.scatter(X2_test["mean radius"], X2_test["mean texture"], c=y2_test, edgecolor="k")
plt.xlabel("Mean radius")
plt.ylabel("Mean texture")
plt.title("Logistic regression probability surface")
plt.show()
```

The contour lines are the model's probabilities; the line where the color is
neutral is the 0.5 decision boundary. Because logistic regression is linear in the
features, that boundary is a straight line.

Hint for question 4:

```python
from sklearn.preprocessing import PolynomialFeatures

poly_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
).fit(X_train, y_train)

poly_proba = poly_model.predict_proba(X_test)[:, 1]
print("Polynomial test accuracy:", round(accuracy_score(y_test, poly_model.predict(X_test)), 3))
print("Polynomial test log loss:", round(log_loss(y_test, poly_proba), 3))
```

Compare these with the plain model's 0.965 accuracy and 0.077 log loss. A more
flexible model can fit a curved boundary, but with 30 correlated features it can
also fit noise.

## 14. What We Learned

- Logistic regression computes a linear score and turns it into a probability with
  the sigmoid.
- A probability becomes a label only after we choose a threshold.
- Log loss measures the quality of probabilities and punishes confident mistakes.
- The gradient is `prediction - truth`, averaged over rows and multiplied by the
  features.
- Scaling and regularization matter, just as they did for Ridge and Lasso.
- Accuracy alone is not enough. A 63% accurate baseline can have zero recall for the
  class we care about.
- The confusion matrix, precision, recall, specificity, and F1 tell the real story.
- Lowering the threshold usually increases recall and decreases precision; the right
  balance depends on the cost of mistakes.
- Coefficients are log-odds effects. Exponentiating them gives odds ratios.
- Multiclass logistic regression uses either one-vs-rest or a multinomial model,
  and the two can perform differently.
- Always compare against a baseline on the same held-out rows.

## Outcome

Learners can train and evaluate binary logistic regression, interpret predicted
probabilities and coefficients as odds ratios, read a confusion matrix, compute
precision and recall, choose and justify a threshold, explain log loss, and extend
the model to multiclass classification with honest held-out evaluation.
