# Day 20 — Practical: Pipelines, Hyperparameter Tuning, and Responsible ML

**Module:** Module 5: Evaluation and Optimization

In this practical we will:

- Build a mixed-type dataset with numeric and categorical columns and missing values.
- Construct a `Pipeline` with a `ColumnTransformer`.
- Evaluate an untuned model with cross-validation.
- Run `GridSearchCV` on a linear model and on a decision tree.
- Run `RandomizedSearchCV` and compare it with grid search.
- Read feature importances and evaluate accuracy by subgroup.
- Demonstrate target-encoding leakage when preprocessing is done outside the folds.

The dataset is generated, so nothing is downloaded.

## 1. Setup

```python
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score

RANDOM_STATE = 42
```

## 2. Build a Mixed-Type Dataset

We generate six numeric signals, then derive two categorical columns and inject
missing values. The target is a synthetic "churn" label.

```python
X_base, y = make_classification(
    n_samples=1200,
    n_features=6,
    n_informative=3,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=0.8,
    flip_y=0.1,
    random_state=RANDOM_STATE,
)

columns = [
    "tenure",
    "monthly_charge",
    "support_calls",
    "usage_hours",
    "age",
    "satisfaction",
]
df = pd.DataFrame(X_base, columns=columns)

df["plan"] = np.where(df["satisfaction"] > -0.1, "premium", "basic")
df["region"] = pd.cut(
    df["age"], bins=3, labels=["north", "south", "west"]
).astype(object)

rng = np.random.default_rng(0)
for column in ["monthly_charge", "support_calls", "region"]:
    rows = rng.choice(df.index, size=60, replace=False)
    df.loc[rows, column] = np.nan

y = pd.Series(y, name="churn")

print("shape:", df.shape)
print(df.isna().sum())
print("target counts:", y.value_counts().to_dict())
```

Expected output:

```text
shape: (1200, 8)
tenure             0
monthly_charge    60
support_calls     60
usage_hours        0
age                0
satisfaction       0
plan               0
region            60
dtype: int64
target counts: {1: 603, 0: 597}
```

There are 60 missing values in each of two numeric columns and in `region`. A naive
model would either crash or need separate handling per column.

## 3. Build the Preprocessing Pipeline

```python
numeric_features = [
    "tenure",
    "monthly_charge",
    "support_calls",
    "usage_hours",
    "age",
    "satisfaction",
]
categorical_features = ["plan", "region"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])
```

Check the shape after fitting the preprocessing on the training data:

```python
df_train, df_test, y_train, y_test = train_test_split(
    df, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

preprocess.fit(df_train)
print("transformed shape:", preprocess.transform(df_train).shape)
```

Expected output:

```text
transformed shape: (900, 11)
```

Six numeric features become six scaled columns. `plan` has two categories and `region`
has three, so the encoder produces five one-hot columns. Eleven columns total.

## 4. Untuned Logistic Regression

```python
logistic_pipeline = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=1000)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

scores = cross_val_score(logistic_pipeline, df, y, cv=cv, scoring="accuracy")
print("fold scores:", np.round(scores, 4))
print("mean:", round(scores.mean(), 4))
print("std:", round(scores.std(), 4))

logistic_pipeline.fit(df_train, y_train)
print("test accuracy:", round(logistic_pipeline.score(df_test, y_test), 4))
```

Expected output:

```text
fold scores: [0.7625 0.7625 0.7292 0.7667 0.7167]
mean: 0.7475
std: 0.0205
test accuracy: 0.7533
```

The cross-validation mean and the test score agree closely, which is a good sign that
the pipeline is being evaluated honestly.

Note the reuse of `preprocess` in the pipeline. When the pipeline is fitted, it clones
and refits the preprocessing inside each fold. The earlier manual `fit` does not carry
over.

## 5. Grid Search on Logistic Regression

```python
param_grid = {"classifier__C": [0.01, 0.1, 1, 10]}

grid = GridSearchCV(
    logistic_pipeline,
    param_grid,
    cv=cv,
    scoring="accuracy",
)

grid.fit(df_train, y_train)

print("best params:", grid.best_params_)
print("best CV score:", round(grid.best_score_, 4))
print("test accuracy:", round(grid.score(df_test, y_test), 4))
```

Expected output:

```text
best params: {'classifier__C': 0.1}
best CV score: 0.7389
test accuracy: 0.7533
```

Tuning did not help. The default `C=1` and the chosen `C=0.1` give the same test
accuracy, and the best CV score is slightly below the untuned mean. This is a common
and useful result: for a well-scaled linear model on moderate data, the regularization
path is flat. Tuning is not guaranteed to improve a model.

The lesson is not that tuning is useless. It is that the search must be judged by the
same honest protocol, and sometimes the honest answer is "keep the default."

## 6. Tune a Decision Tree

Decision trees are far more sensitive to depth and leaf size, so tuning has more room
to help.

```python
tree_pipeline = Pipeline([
    ("preprocess", preprocess),
    ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
])

scores = cross_val_score(tree_pipeline, df, y, cv=cv, scoring="accuracy")
print("untuned folds:", np.round(scores, 4))
print("untuned mean:", round(scores.mean(), 4))

tree_pipeline.fit(df_train, y_train)
print("untuned test:", round(tree_pipeline.score(df_test, y_test), 4))
```

Expected output:

```text
untuned folds: [0.7833 0.8167 0.825  0.8417 0.775 ]
untuned mean: 0.8083
untuned test: 0.7967
```

Now search over depth and minimum leaf size:

```python
tree_grid = {
    "classifier__max_depth": [2, 3, 4, 5, 6, 8, None],
    "classifier__min_samples_leaf": [1, 5, 10, 20],
}

grid_tree = GridSearchCV(
    tree_pipeline,
    tree_grid,
    cv=cv,
    scoring="accuracy",
)

grid_tree.fit(df_train, y_train)

print("best params:", grid_tree.best_params_)
print("best CV score:", round(grid_tree.best_score_, 4))
print("tuned test:", round(grid_tree.score(df_test, y_test), 4))
```

Expected output:

```text
best params: {'classifier__max_depth': 6, 'classifier__min_samples_leaf': 20}
best CV score: 0.8433
tuned test: 0.8
```

Tuning raised cross-validation accuracy from 0.8083 to 0.8433, about 3.5 points. The
test score moved from 0.7967 to 0.80, a smaller gain, which is typical: the search
optimized the CV folds, and some of that improvement is fold-specific.

This is exactly why the test set must stay untouched until the end.

## 7. Randomized Search

A randomized search over the same tree family samples a fixed number of
configurations.

```python
from scipy.stats import randint

param_distributions = {
    "classifier__max_depth": randint(2, 12),
    "classifier__min_samples_leaf": randint(1, 25),
}

random_search = RandomizedSearchCV(
    tree_pipeline,
    param_distributions,
    n_iter=20,
    cv=cv,
    scoring="accuracy",
    random_state=RANDOM_STATE,
)

random_search.fit(df_train, y_train)

print("best params:", random_search.best_params_)
print("best CV score:", round(random_search.best_score_, 4))
print("test:", round(random_search.score(df_test, y_test), 4))
```

Expected output:

```text
best params: {'classifier__max_depth': 8, 'classifier__min_samples_leaf': 20}
best CV score: 0.8433
test: 0.8
```

Twenty random samples matched the full grid's cross-validation score and test score.
Here the grid had 28 combinations, so random search was only slightly cheaper, but on
a larger space the savings grow quickly.

## 8. Feature Importances

The tuned tree exposes feature importances through the pipeline.

```python
best_tree = grid_tree.best_estimator_

importances = best_tree.named_steps["classifier"].feature_importances_
names = best_tree.named_steps["preprocess"].get_feature_names_out()

importance_table = (
    pd.Series(importances, index=names)
    .sort_values(ascending=False)
    .round(3)
)
print(importance_table)
```

Expected output:

```text
num__monthly_charge    0.490
num__tenure            0.393
num__age               0.110
num__usage_hours       0.006
num__support_calls     0.001
num__satisfaction      0.000
cat__plan_basic        0.000
cat__plan_premium      0.000
cat__region_north      0.000
cat__region_south      0.000
cat__region_west       0.000
```

`monthly_charge` and `tenure` dominate. The derived categorical columns contribute
little here, which makes sense because `plan` was derived from `satisfaction` and
`region` from `age`, and neither carries much extra signal.

Feature importance describes how the tree used a feature, not whether the feature
causes the outcome. Correlated features can share or mask importance.

## 9. Evaluate by Subgroup

Aggregate accuracy can hide unequal performance. Compare the two plan groups.

```python
predictions = best_tree.predict(df_test)

audit = df_test.copy()
audit["actual"] = y_test.values
audit["predicted"] = predictions

for group, subset in audit.groupby("plan"):
    print(
        f"plan={group:8s} n={len(subset):3d} "
        f"accuracy={accuracy_score(subset['actual'], subset['predicted']):.4f} "
        f"recall={recall_score(subset['actual'], subset['predicted'], zero_division=0):.4f}"
    )
```

Expected output:

```text
plan=basic    n=141 accuracy=0.8156 recall=0.7826
plan=premium  n=159 accuracy=0.7862 recall=0.7439
```

The premium group has slightly lower accuracy and recall. The gap is small here, but
the exercise is the point: subgroup metrics are how unfairness becomes visible. On a
real system, repeat this for every protected or operationally important group, and
decide whether the gap is acceptable.

## 10. Leakage from Target Encoding Outside the Folds

Target encoding replaces a category with the mean target for that category. It is
powerful for high-cardinality features, and dangerous if computed on all rows before
cross-validation.

Create a random high-cardinality category that has no real relationship with the
target.

```python
rng = np.random.default_rng(1)
n = 1000

category = pd.Series([f"id_{i}" for i in rng.integers(0, 700, n)])
target = pd.Series(rng.integers(0, 2, n))

# WRONG: the encoding uses each row's own target
encoding = category.map(target.groupby(category).mean())

leaky_scores = cross_val_score(
    LogisticRegression(max_iter=1000),
    encoding.values.reshape(-1, 1),
    target,
    cv=5,
)
print("leaky target encoding:", round(leaky_scores.mean(), 4))
```

Expected output:

```text
leaky target encoding: 0.798
```

The category is random, so the honest score should be near 0.5. Instead it is 0.798,
because each category's encoding was computed partly from the row's own label.

Now do it correctly: one-hot encode inside a pipeline so the encoder is fit only on
each training fold.

```python
honest_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ("classifier", LogisticRegression(max_iter=1000)),
])

honest_scores = cross_val_score(
    honest_pipeline,
    category.values.reshape(-1, 1),
    target,
    cv=5,
)
print("honest one-hot pipeline:", round(honest_scores.mean(), 4))
```

Expected output:

```text
honest one-hot pipeline: 0.532
```

The honest estimate is 0.532, close to chance. The 27-point gap is pure leakage. Any
transformation that uses the target must be fit inside the fold, or it will produce a
model that looks good and fails in production.

The same rule applies to `SelectKBest`, PCA, scaling, imputation, and resampling.

## 11. Try It Yourself

1. **Change an imputation strategy.** Tune `preprocess__num__imputer__strategy` over
   `["mean", "median"]`. Does the choice matter here?
2. **Add a missing indicator.** Use `SimpleImputer(add_indicator=True)` and see whether
   the "was missing" flags help.
3. **Tune the whole pipeline.** Include the numeric imputer strategy and the tree
   depth in one `param_grid`. Confirm the names resolve correctly.
4. **Compare search budgets.** Time a 100-point random search against a grid of 100
   combinations. Which covers more useful configurations?
5. **Use nested CV.** Wrap `GridSearchCV` inside `cross_val_score` and compare with
   `best_score_`. How large is the optimism?
6. **Audit another group.** Evaluate by `region` and by missingness of
   `monthly_charge`. Are any groups systematically worse?
7. **Handle imbalance.** Add `class_weight="balanced"` to the tree and compare
   subgroup recall.

## 12. What We Learned

- A `Pipeline` fits preprocessing and the model as one unit.
- `ColumnTransformer` applies different steps to numeric and categorical columns.
- `handle_unknown="ignore"` protects against unseen categories at prediction time.
- Cross-validation of the pipeline is honest because each fold refits everything.
- Tuning a linear model on this data did not help; tuning a tree did.
- Randomized search can match grid search with fewer evaluations.
- Subgroup metrics expose performance gaps that aggregate accuracy hides.
- Target encoding computed outside the folds leaks and inflates scores.
- All target-dependent preprocessing belongs inside the pipeline.

## Outcome

Learners can build a `Pipeline` with `ColumnTransformer`, apply imputation, encoding,
and scaling correctly, run grid and randomized hyperparameter search inside
cross-validation, interpret the difference between tuned and untuned models, inspect
feature importance, audit subgroup performance, and prevent preprocessing leakage.
