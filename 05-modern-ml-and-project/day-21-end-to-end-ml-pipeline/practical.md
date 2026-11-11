# Day 21 — Practical: End-to-End ML Pipeline

**Module:** Module 6: Modern ML and Project

In this practical we will:

- Define a regression problem and inspect a real CSV.
- Split the data before any evaluation.
- Build a leak-free pipeline with `ColumnTransformer`.
- Establish a trivial baseline and a simple model baseline.
- Compare two improved models with cross-validation.
- Select a model on validation data and evaluate it once on the test set.
- Save the complete pipeline with `joblib`, reload it, and predict new records.

The dataset is `00-foundations/day-04-bias-variance-ml-workflow/house_prices.csv`.

## 1. Setup

```python
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

RANDOM_STATE = 42
```

## 2. Load and Inspect the Data

```python
path = "../../00-foundations/day-04-bias-variance-ml-workflow/house_prices.csv"
df = pd.read_csv(path)

print(df.shape)
print(df.dtypes)
print(df.isna().sum())
print(df.head())
```

Expected output (abridged):

```text
(140, 4)
area_sqft      int64
age_years    float64
location      object
price_usd      int64
dtype: object
area_sqft     0
age_years     6
location      0
price_usd     0
dtype: int64
```

The dataset has 140 houses, two numeric features, one categorical feature, one target,
and six missing values in `age_years`.

## 3. Define the Problem

Write the problem statement before modeling:

> Given a house's area, age, and location, predict its sale price so that a listing
> tool can flag prices that look unrealistic.

- **Unit**: one house.
- **Target**: `price_usd`, continuous.
- **Features available at prediction time**: `area_sqft`, `age_years`, `location`.
- **Metric**: root mean squared error (RMSE) in dollars, with MAE and R² as secondary
  metrics.
- **Baseline**: predict the training mean.
- **Constraint**: the model must be easy to retrain and cheap to serve.

All three features would be known at listing time, so there is no obvious temporal
leakage.

## 4. Split Before Evaluating

```python
X = df.drop(columns="price_usd")
y = df["price_usd"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print("train:", X_train.shape, "test:", X_test.shape)
```

Expected output:

```text
train: (112, 3) test: (28, 3)
```

The test set stays untouched until the selected model is final.

## 5. Build the Preprocessing Pipeline

```python
numeric_features = ["area_sqft", "age_years"]
categorical_features = ["location"]

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

The median imputer fills the six missing ages, the scaler puts area and age on
comparable ranges, and the encoder turns `location` into binary columns. All of it is
fit inside each training fold.

## 6. Baselines

A trivial baseline predicts the training mean. A simple model baseline uses linear
regression on the preprocessed features.

```python
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

dummy = DummyRegressor(strategy="mean")
linear = Pipeline([
    ("preprocess", preprocess),
    ("regressor", LinearRegression()),
])

for name, model in [("dummy", dummy), ("linear", linear)]:
    scores = cross_val_score(
        model, X_train, y_train, cv=cv,
        scoring="neg_root_mean_squared_error",
    )
    rmse = -scores
    print(
        f"{name:8s} CV RMSE={rmse.mean():.2f} "
        f"std={rmse.std():.2f}"
    )
```

Expected output:

```text
dummy    CV RMSE=56504.03 std=3850.27
linear   CV RMSE=11423.07 std=1166.01
```

The dummy predicts the same value for every house and is off by about $56,500 on
average. Linear regression cuts the error to about $11,400, a large improvement. That
is the power of even a simple model.

## 7. Improved Models

Try two ensembles that can capture nonlinear relationships and interactions.

```python
random_forest = Pipeline([
    ("preprocess", preprocess),
    ("regressor", RandomForestRegressor(
        n_estimators=300, random_state=RANDOM_STATE)),
])

gradient_boosting = Pipeline([
    ("preprocess", preprocess),
    ("regressor", GradientBoostingRegressor(random_state=RANDOM_STATE)),
])

for name, model in [
    ("random forest", random_forest),
    ("gradient boost", gradient_boosting),
]:
    scores = cross_val_score(
        model, X_train, y_train, cv=cv,
        scoring="neg_root_mean_squared_error",
    )
    rmse = -scores
    print(
        f"{name:14s} CV RMSE={rmse.mean():.2f} "
        f"std={rmse.std():.2f}"
    )
```

Expected output:

```text
random forest  CV RMSE=9927.16 std=999.32
gradient boost CV RMSE=10506.89 std=837.76
```

Both ensembles beat linear regression on cross-validation. Random forest has the best
mean, though its spread is slightly larger than gradient boosting's.

### Compare all candidates

```python
results = {
    "dummy": 56504.03,
    "linear": 11423.07,
    "random forest": 9927.16,
    "gradient boost": 10506.89,
}

for name, rmse in results.items():
    print(f"{name:14s} {rmse:10.2f}")
```

This small table is the model-selection record. Keep it with the experiment.

## 8. Select and Evaluate Once

Select by cross-validation: random forest.

```python
best_model = random_forest
best_model.fit(X_train, y_train)

predictions = best_model.predict(X_test)

rmse = mean_squared_error(y_test, predictions) ** 0.5
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("test RMSE:", round(rmse, 2))
print("test MAE:", round(mae, 2))
print("test R2:", round(r2, 4))
```

Expected output:

```text
test RMSE: 11230.98
test MAE: 8890.18
test R2: 0.9625
```

The test RMSE of about $11,200 is a little higher than the CV estimate of $9,900. With
only 28 test houses, this difference is normal sampling noise. The model explains
about 96% of the variance in the held-out prices.

This is the only test evaluation in the workflow. Do not now go back and try ten more
models on the same test set; that would turn it into a validation set and make the
estimate optimistic.

## 9. Save the Complete Pipeline

Save the pipeline, not just the regressor, so preprocessing travels with the model.

```python
model_path = "house_price_pipeline.joblib"
joblib.dump(best_model, model_path)
print("saved:", model_path)
```

Expected output:

```text
saved: house_price_pipeline.joblib
```

Reload it in a fresh object and confirm it reproduces the same predictions.

```python
loaded_model = joblib.load(model_path)

same = np.allclose(loaded_model.predict(X_test), predictions)
print("predictions match:", same)
```

Expected output:

```text
predictions match: True
```

## 10. Predict New Records

New data arrives without a target and may contain missing values or unseen
categories. The pipeline handles both.

```python
new_houses = pd.DataFrame([
    {"area_sqft": 1500, "age_years": 10.0, "location": "Suburb"},
    {"area_sqft": 2500, "age_years": np.nan, "location": "City"},
    {"area_sqft": 900, "age_years": 30.0, "location": "Town"},
])

new_predictions = loaded_model.predict(new_houses)
print(np.round(new_predictions).astype(int))
```

Expected output:

```text
[ 74278 181720  43005]
```

The second record has a missing age and a `City` location. The imputer fills the age,
and the one-hot encoder produces the columns it learned during training. A manually
built preprocessing script would need separate handling for both cases.

Present the result with context:

```python
for house, price in zip(new_houses.to_dict("records"), new_predictions):
    print(f"{house} -> ${price:,.0f}")
```

## 11. Record Reproducibility Metadata

A saved file without context is hard to trust. Write a small metadata record next to
it.

```python
metadata = {
    "model": "RandomForestRegressor(n_estimators=300)",
    "target": "price_usd",
    "features": ["area_sqft", "age_years", "location"],
    "cv_rmse": 9927.16,
    "test_rmse": 11230.98,
    "random_state": RANDOM_STATE,
    "data_rows": len(df),
    "training_rows": len(X_train),
}

print(metadata)
```

In a real project, store this in a file or experiment tracker together with the data
snapshot, code commit, and package versions. The model file alone is not enough to
reproduce the result.

## 12. Try It Yourself

1. **Tune the forest.** Use `GridSearchCV` over `n_estimators` and `max_depth` on the
   training set, then evaluate once on test.
2. **Add a feature.** Create `age_squared` or `price_per_sqft`-style ratios and see
   whether CV improves.
3. **Compare with a median baseline.** Change `DummyRegressor` to
   `strategy="median"` and explain the difference.
4. **Break the pipeline.** Remove `handle_unknown="ignore"` and predict a location
   that was not in training. What happens?
5. **Add metadata.** Save the metadata dictionary as JSON and load it with the model.
6. **Simulate drift.** Multiply every test `area_sqft` by 1.3 and compare RMSE before
   and after. What does this suggest about monitoring?
7. **Use a different metric.** Select by MAE instead of RMSE. Does the winner change?

## 13. What We Learned

- The lifecycle starts with a problem statement and a metric, not an algorithm.
- Split the data before evaluating anything.
- Preprocessing belongs inside the pipeline so training and serving agree.
- A trivial baseline turns a score into a meaningful comparison.
- Linear regression was a strong simple baseline; ensembles improved on it.
- Model selection belongs on validation data; the test set is evaluated once.
- The full pipeline can be saved with `joblib` and reproduces predictions exactly.
- The pipeline handles missing values and unseen categories in new records.
- Metadata is part of the model artifact.

## Outcome

Learners can define a regression problem, split data before evaluation, build a
leak-free preprocessing and model pipeline, compare baselines and improved models with
cross-validation, evaluate once on a held-out test set, save and reload the complete
pipeline, generate predictions for new records, and record reproducibility metadata.
