# Day 04 — Practical: Bias, Variance, and a Complete Workflow

**Module:** Module 1: Foundations

In this practical we will use a small house-price dataset to see bias and
variance with our own eyes. We will:

- Train a model that is too simple and watch it underfit.
- Train a model that is too complex and watch it overfit.
- Find a balanced model.
- Build a safe preprocessing and training pipeline.
- Discuss how the same ideas apply to a real project.

We will use `house_prices.csv`, a synthetic dataset of 140 houses. It has three
features and one target.

| Column | Meaning | Type |
|--------|---------|------|
| `area_sqft` | Size of the house in square feet | Number |
| `age_years` | Age of the house in years | Number (6 missing) |
| `location` | `City`, `Suburb`, or `Town` | Category |
| `price_usd` | Selling price in US dollars | Target |

The data is synthetic, but the pattern is realistic: bigger houses cost more,
and the price grows faster for very large houses. Older houses and location also
have a small effect.

## 1. Setup

Start Jupyter from the repository folder and create a notebook inside
`00-foundations/day-04-bias-variance-ml-workflow/` (see the
[Day 01 practical](../day-01-introduction-to-ml/practical.md) if you need a
reminder).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

RANDOM_STATE = 42
print("Libraries loaded")
```

Why a constant for the random seed? Because we want every run to use the same
split and the same model randomness. This makes the lesson reproducible. If you
change the seed, the exact numbers will change a little.

## 2. Load and Inspect the Data

```python
df = pd.read_csv("house_prices.csv")
df.head()
```

If you get a `FileNotFoundError`, the notebook is not in the same folder as the
CSV file. Run Jupyter from this folder, or use the full path.

Now look at the shape, missing values, and basic statistics:

```python
print("Shape:", df.shape)
print(df.isnull().sum())
df.describe().round(1)
```

Expected output (approximately):

```
Shape: (140, 4)
area_sqft    0
age_years    6
location     0
price_usd    0
dtype: int64
```

There are **140 houses** and **6 missing ages**. The target `price_usd` ranges
from about **16,000** to **238,000**.

Also check the categories:

```python
df["location"].value_counts()
```

Expected output:

```
City      55
Suburb    49
Town      36
```

The classes are reasonably balanced, so no location is extremely rare.

## 3. Define Features, Target, and Split

```python
X = df.drop(columns=["price_usd"])
y = df["price_usd"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
```

Expected output:

```
Training rows: 112
Test rows: 28
```

The model learns from 112 houses and is tested on 28 houses it has never seen.

Next, define a helper that measures error:

```python
def rmse(actual, predicted):
    return mean_squared_error(actual, predicted) ** 0.5
```

**RMSE** means **root mean squared error**. It is in the same units as the
target, so here it is measured in dollars. A lower RMSE is better. For example,
an RMSE of 10,000 means predictions are typically off by about 10,000 dollars.

## 4. The Baseline: Always Predict the Average

Before training a real model, we need a simple baseline. The simplest possible
model ignores all features and always predicts the average training price.

```python
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train[["area_sqft"]], y_train)

train_rmse = rmse(y_train, baseline.predict(X_train[["area_sqft"]]))
test_rmse = rmse(y_test, baseline.predict(X_test[["area_sqft"]]))

print("Baseline train RMSE:", round(train_rmse))
print("Baseline test RMSE:", round(test_rmse))
```

Expected output:

```
Baseline train RMSE: 56420
Baseline test RMSE: 58015
```

The two errors are high and close together. The model is consistently wrong
because it ignores the most important information: house size. This is **high
bias**. It is **underfitting**.

Any useful model should beat this baseline.

## 5. A Model That Is Too Simple

For a clear picture, let us use only **one feature**: `area_sqft`. A tree with
`max_depth=1` can ask only one yes/no question.

```python
area_train = X_train[["area_sqft"]]
area_test = X_test[["area_sqft"]]

simple_model = DecisionTreeRegressor(max_depth=1, random_state=RANDOM_STATE)
simple_model.fit(area_train, y_train)

simple_train = rmse(y_train, simple_model.predict(area_train))
simple_test = rmse(y_test, simple_model.predict(area_test))

print("Simple train RMSE:", round(simple_train))
print("Simple test RMSE:", round(simple_test))
```

Expected output:

```
Simple train RMSE: 28533
Simple test RMSE: 29158
```

This is much better than the baseline, so area clearly matters. But the errors
are still high, and they are close together. The model is still underfitting.

Let us see the one question it learned:

```python
print(export_text(simple_model, feature_names=["area_sqft"], decimals=0))
```

Expected output:

```
|--- area_sqft <= 1950
|   |--- value: [69267]
|--- area_sqft >  1950
|   |--- value: [166676]
```

In plain words:

- If the house is **1,950 sqft or smaller**, predict **69,267**.
- If the house is **larger than 1,950 sqft**, predict **166,676**.

This is like sorting houses into two buckets. It cannot follow the smooth upward
curve, so it consistently underprices very large houses and misprices many small
ones. That is **bias**: the model is too simple for the real pattern.

## 6. Add Complexity Step by Step

Now let us allow deeper trees. A deeper tree can ask more questions and fit more
detail. We will compare the training and test RMSE for several depths.

```python
depths = [1, 2, 3, 4, 5, 6, 8, None]
rows = []

for depth in depths:
    model = DecisionTreeRegressor(max_depth=depth, random_state=RANDOM_STATE)
    model.fit(area_train, y_train)

    train = rmse(y_train, model.predict(area_train))
    test = rmse(y_test, model.predict(area_test))

    rows.append(
        {
            "max_depth": "None" if depth is None else depth,
            "train_rmse": round(train),
            "test_rmse": round(test),
            "gap": round(test - train),
        }
    )

results = pd.DataFrame(rows)
results
```

Expected result (values may differ slightly by scikit-learn version):

| max_depth | train_rmse | test_rmse | gap |
|-----------|-----------:|----------:|----:|
| 1 | 28,533 | 29,158 | 625 |
| 2 | 15,062 | 19,708 | 4,646 |
| 3 | 9,183 | 15,155 | 5,972 |
| 4 | 7,319 | 11,497 | 4,178 |
| 5 | 6,076 | 13,295 | 7,219 |
| 6 | 4,692 | 13,237 | 8,545 |
| 8 | 2,753 | 13,287 | 10,534 |
| None | 1,380 | 12,961 | 11,581 |

Read the table from top to bottom:

- **Depth 1:** high training error and high test error, close together. **High
  bias, underfitting.**
- **Depth 4:** the lowest test error, with a reasonably low training error.
  **Good balance.**
- **Depth 5 and deeper:** training error keeps falling, but test error stops
  improving and then gets worse. The gap grows. **High variance, overfitting.**
- **Unlimited (`None`):** training error is almost zero, but test error is worse
  than depth 4. The model memorized the training houses.

> The best model is not the one with the lowest training error. Depth 4 wins
> here because it has the best **test** error.

> Note: for this small lesson we compare depths on the test set. In a real
> project, compare them on a **validation set** and keep the test set for one
> final check at the end.

## 7. Visualize Underfitting, Good Fit, and Overfitting

Numbers are useful, but a picture makes the tradeoff obvious. We will plot the
training houses and the prediction of three trees.

```python
def plot_fit(depth, label, ax):
    model = DecisionTreeRegressor(max_depth=depth, random_state=RANDOM_STATE)
    model.fit(area_train, y_train)

    grid = pd.DataFrame(
        {
            "area_sqft": np.linspace(
                area_train["area_sqft"].min(),
                area_train["area_sqft"].max(),
                300,
            )
        }
    )

    ax.scatter(
        area_train["area_sqft"],
        y_train,
        s=12,
        alpha=0.35,
        label="Training houses",
    )
    ax.plot(grid["area_sqft"], model.predict(grid), linewidth=2, label=label)
    ax.set_xlabel("Area (sqft)")
    ax.set_ylabel("Price (USD)")
    ax.legend()


fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

plot_fit(1, "Depth 1 (underfit)", axes[0])
axes[0].set_title("Underfit: too simple")

plot_fit(4, "Depth 4 (good fit)", axes[1])
axes[1].set_title("Good fit: follows the trend")

plot_fit(None, "Unlimited (overfit)", axes[2])
axes[2].set_title("Overfit: memorizes noise")

plt.tight_layout()
plt.show()
```

What to look for:

- **Depth 1:** two flat steps. The model cannot follow the curve, so it is biased.
- **Depth 4:** several steps that roughly follow the upward trend.
- **Unlimited:** many tiny steps that jump up and down to touch individual
  houses. It is fitting noise, not just the trend.

The unlimited tree looks impressive on the training points. That is exactly the
trap: it is **too good on data it has already seen**.

## 8. One Prediction Does Not Tell the Whole Story

Let us ask all three models about the same house: 1,800 sqft.

```python
new_house = pd.DataFrame({"area_sqft": [1800]})

for depth in [1, 4, None]:
    model = DecisionTreeRegressor(max_depth=depth, random_state=RANDOM_STATE)
    model.fit(area_train, y_train)
    prediction = model.predict(new_house)[0]
    print(f"max_depth={depth}: {round(prediction):,}")
```

Expected output:

```
max_depth=1: 69,267
max_depth=4: 98,964
max_depth=None: 101,000
```

The simple model is far below the others. For this one house, the complex models
look better. But remember: on the **whole test set**, the unlimited tree was
worse than depth 4. One prediction can be lucky or unlucky.

This is why we compare models on a whole validation or test set, not on a single
example.

## 9. Build a Safe Preprocessing and Training Pipeline

Now let us use **all** the features. This introduces two practical issues:

1. `age_years` has missing values.
2. `location` is text, but models need numbers.

We could fix these by hand, but it is safer to put every step into a **pipeline**.
A pipeline guarantees that the same transformations are applied to training,
validation, and test data, and that they are fitted on training data only.

```python
numeric_features = ["area_sqft", "age_years"]
categorical_features = ["location"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)
```

What each piece does:

- **`SimpleImputer(strategy="median")`** fills missing ages with the median age
  of the **training** data.
- **`StandardScaler()`** puts numeric features on a similar scale. Trees do not
  need this, but linear models do, and it is a good habit.
- **`OneHotEncoder()`** turns `location` into separate 0/1 columns, such as
  `location_City`, `location_Suburb`, and `location_Town`.
- **`ColumnTransformer`** applies different steps to different columns.
- **`Pipeline`** chains the steps together so they cannot be applied in the wrong
  order.

Now add the model:

```python
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE)),
    ]
)

pipeline.fit(X_train, y_train)

pipeline_train = rmse(y_train, pipeline.predict(X_train))
pipeline_test = rmse(y_test, pipeline.predict(X_test))

print("Pipeline train RMSE:", round(pipeline_train))
print("Pipeline test RMSE:", round(pipeline_test))
```

Expected output:

```
Pipeline train RMSE: 6970
Pipeline test RMSE: 12076
```

The pipeline automatically:

1. Learned the median age from the training data.
2. Filled the missing age in both sets with that median.
3. Scaled the numeric columns using training statistics.
4. Encoded the locations.
5. Trained the tree on the prepared data.

The result is similar to the area-only depth-4 tree. Adding `age_years` and
`location` did not help much here because area carries most of the signal. That
is a useful lesson: **more features do not always mean a better model.**

Let us check which features the tree used:

```python
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = pipeline.named_steps["model"].feature_importances_

pd.DataFrame(
    {"feature": feature_names, "importance": importances}
).sort_values("importance", ascending=False)
```

Expected result:

| feature | importance |
|---------|-----------:|
| num__area_sqft | 0.996 |
| num__age_years | 0.003 |
| cat__location_City | 0.001 |
| cat__location_Suburb | 0.000 |
| cat__location_Town | 0.000 |

The tree relies almost completely on `area_sqft`. Feature importances are not
proof of causation, and they can change with a different split or seed, but they
are a useful sanity check.

## 10. Data Leakage: The Easy Mistake

A pipeline is not just convenient; it also protects us from **data leakage**. A
common mistake is to fill missing values **before** splitting:

```python
leaky = df.copy()
median_age = leaky["age_years"].median()
leaky["age_years"] = leaky["age_years"].fillna(median_age)

X_leaky = leaky.drop(columns=["price_usd"])
y_leaky = leaky["price_usd"]
X_train_leaky, X_test_leaky, y_train_leaky, y_test_leaky = train_test_split(
    X_leaky, y_leaky, random_state=RANDOM_STATE
)
```

This is wrong because the test set helped choose the median. The model has
already seen a little information from the test data.

The safe order is:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

median_age = X_train["age_years"].median()
X_train["age_years"] = X_train["age_years"].fillna(median_age)
X_test["age_years"] = X_test["age_years"].fillna(median_age)
```

Even better, let the pipeline do it. When `pipeline.fit(X_train, y_train)` runs,
the imputer learns the median from `X_train` only. When `pipeline.predict(X_test)`
runs, it applies that same median to the test data.

The same rule applies to scaling, encoding, feature selection, and any other step
that learns something from the data.

## 11. What Would Change in a Real Project?

Our dataset is small and synthetic. In a real house-price project, we would also:

- **Use a time-based split.** Train on older sales and test on newer sales,
  because the housing market changes over time. A random split can leak future
  information.
- **Check duplicate or related rows.** The same house sold twice, or several
  photos of the same house, should not appear in both training and test sets.
- **Investigate outliers.** A house sold between relatives at an unusual price
  can mislead the model.
- **Add better features.** Bedrooms, bathrooms, condition, school district, and
  distance to the city center often matter more than age.
- **Choose the metric carefully.** RMSE punishes large errors. If percentage
  error matters more, use a percentage-based metric.
- **Keep the test set untouched.** Use validation data for tuning, and open the
  test set once at the end.
- **Monitor after deployment.** House prices change, so the model must be
  retrained regularly.

The same workflow applies to employee attrition:

- Define the decision: who should receive support?
- Remove employee IDs and information collected after the employee left.
- Split by employee, not by row.
- Compare a baseline, a simple model, and a complex model.
- Choose by validation performance, not training performance.

## 12. Try It Yourself

Use the `X`, `y`, `X_train`, `X_test`, `area_train`, and `area_test` variables
from this notebook.

1. Change the split to `test_size=0.3` and `random_state=0`. How do the RMSE
   values change?
2. Try `max_depth=2` and `max_depth=8`. Which gives the best test RMSE now?
3. Train a model using only `age_years`. Is it underfitting or overfitting?
   Explain why.
4. Train an unlimited tree on all features. Compare its training and test RMSE
   with the depth-4 pipeline. Which one would you trust?
5. Replace `SimpleImputer(strategy="median")` with `strategy="mean"`. Does the
   test RMSE change much?
6. Replace the tree in the pipeline with `Ridge()`. Does a regularized linear
   model do better or worse?

Hints:

```python
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y, test_size=0.3, random_state=0
)

age_model = DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE)
age_model.fit(X_train[["age_years"]].fillna(X_train["age_years"].median()), y_train)

from sklearn.linear_model import Ridge

ridge_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=10.0)),
    ]
)
ridge_pipeline.fit(X_train, y_train)
print(round(rmse(y_test, ridge_pipeline.predict(X_test))))
```

For question 4, remember that training RMSE near zero is a warning sign, not a
win.

## 13. What We Learned

In this practical we:

- Built a baseline that always predicts the average price.
- Saw a depth-1 tree underfit because it was too simple.
- Saw a deep tree overfit because it memorized training noise.
- Found a balanced tree depth by comparing training and test RMSE.
- Visualized underfitting, a good fit, and overfitting.
- Built a preprocessing and training pipeline with imputation, scaling,
  one-hot encoding, and a model.
- Learned why preprocessing must be fitted on the training set only.
- Discussed how the same workflow applies to real projects.

## Outcome

Learners can explain bias and variance using a concrete example, recognize
underfitting and overfitting from training and test error, choose model
complexity using validation or test performance, and build a safe preprocessing
and training pipeline.
