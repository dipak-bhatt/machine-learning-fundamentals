# Day 05 — Practical: Your First Regression Line

**Module:** Module 2: Supervised Learning

In this practical we will use the house-price dataset from Day 04 to build our
first regression model. We will:

- Look at the relationship between house area and price.
- Fit a simple linear regression line.
- Read the slope and intercept.
- Make predictions for new houses.
- Inspect residuals.
- Calculate MAE, MSE, RMSE, and R².
- Plot the fit and the errors.
- Compare simple regression with multiple regression.

The dataset is the same one we used in Day 04:
`../../00-foundations/day-04-bias-variance-ml-workflow/house_prices.csv`.

| Column | Meaning | Type |
|--------|---------|------|
| `area_sqft` | Size of the house in square feet | Number |
| `age_years` | Age of the house in years | Number (6 missing) |
| `location` | `City`, `Suburb`, or `Town` | Category |
| `price_usd` | Selling price in US dollars | Target |

## 1. Setup

Start Jupyter from the repository folder and create a notebook inside
`01-supervised-learning/day-05-linear-regression/` (see the
[Day 01 practical](../../00-foundations/day-01-introduction-to-ml/practical.md)
if you need a reminder).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
print("Libraries loaded")
```

`RANDOM_STATE` keeps the train/test split repeatable. With the same seed, you and
I get the same result.

## 2. Load and Inspect the Data

```python
df = pd.read_csv("../../00-foundations/day-04-bias-variance-ml-workflow/house_prices.csv")
df.head()
```

If you get a `FileNotFoundError`, your notebook is not inside the Day 05 folder.
The path above goes up two folders to the repository root, then into the Day 04
folder. Either move the notebook or use the full path to the CSV.

Now check the shape, missing values, and summary statistics:

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

The dataset has **140 houses**. Only `age_years` has missing values, and we will
not use age in the first model, so the simple regression can use all 140 rows.

The price ranges from about **16,000** to **238,000**, with an average of about
**116,429**.

## 3. Look at the Relationship

Before fitting a model, always look at the data. A scatter plot shows whether a
straight line is a reasonable idea.

```python
plt.figure(figsize=(8, 5))
plt.scatter(df["area_sqft"], df["price_usd"], alpha=0.5)
plt.xlabel("Area (sqft)")
plt.ylabel("Price (USD)")
plt.title("House area vs. price")
plt.show()
```

What to look for:

- The points generally rise from left to right, so area and price are positively
  associated.
- The cloud is not perfectly straight. Price seems to grow faster for very large
  houses.
- A few houses sit far above or below the main cloud. These may be unusual cases
  or simply noise.

A straight line can still be a useful first model, even when the pattern is not
perfect. We will check the residuals later to see what the line misses.

## 4. Define Features, Target, and Split

For **simple** linear regression we use one feature:

```python
X = df[["area_sqft"]]
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

Two details matter:

- `X = df[["area_sqft"]]` keeps `X` as a **2D table** with one column. If you
  write `df["area_sqft"]`, you get a 1D Series, and scikit-learn will complain.
- We split **before** fitting anything, so the test houses stay unseen.

## 5. Fit the Simple Linear Regression

```python
model = LinearRegression()
model.fit(X_train, y_train)

print("Slope (coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)
```

Expected output (approximately):

```
Slope (coefficient): 84.40062148442131
Intercept: -38428.22812817148
```

The model learned this rule:

```
predicted_price = -38,428.23 + 84.40 × area_sqft
```

In plain words:

- **Slope 84.40:** each extra square foot is associated with about **84 dollars**
  more in predicted price.
- **Intercept −38,428.23:** this is the prediction for a house with zero square
  feet. That is not a real house, so we should not take the intercept literally.
  It simply positions the line.

The positive slope matches what we saw in the scatter plot: larger houses tend to
cost more.

## 6. Make Predictions

Let us ask the model about four houses:

```python
new_houses = pd.DataFrame({"area_sqft": [800, 1000, 1800, 2500]})
new_houses["predicted_price"] = model.predict(new_houses).round(0)
new_houses
```

Expected output:

| area_sqft | predicted_price |
|----------:|----------------:|
| 800 | 29,092 |
| 1,000 | 45,972 |
| 1,800 | 113,493 |
| 2,500 | 172,573 |

Check one by hand:

```
-38,428.23 + 84.40 × 1,800 ≈ 113,493
```

All four areas are inside the range of the training data (about 617 to 2,942
sqft), so these are **interpolations**, not wild extrapolations.

## 7. Inspect the Residuals

A residual is `actual − predicted`. Let us look at the training houses:

```python
train_predictions = model.predict(X_train)

residuals = pd.DataFrame(
    {
        "area_sqft": X_train["area_sqft"],
        "actual": y_train,
        "predicted": train_predictions.round(0),
    }
)
residuals["residual"] = residuals["actual"] - residuals["predicted"]
residuals.head(8)
```

Expected output (approximately):

| area_sqft | actual | predicted | residual |
|----------:|-------:|----------:|---------:|
| 1,832 | 117,000 | 116,194 | 806 |
| 2,616 | 187,500 | 182,364 | 5,136 |
| 1,864 | 82,000 | 118,895 | −36,895 |
| 2,474 | 152,500 | 170,379 | −17,879 |
| 1,079 | 49,500 | 52,640 | −3,140 |
| 2,756 | 209,500 | 194,180 | 15,320 |
| 1,067 | 63,500 | 51,627 | 11,873 |
| 2,930 | 231,500 | 208,866 | 22,634 |

How to read this table:

- A **positive** residual means the model predicted **too low**.
- A **negative** residual means the model predicted **too high**.
- The third house is predicted about 36,895 dollars too high. That is a large
  miss, but one house alone does not decide whether the model is good.

## 8. Calculate the Metrics

We need a small helper so we do not repeat the same code:

```python
def evaluate(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = mse ** 0.5
    r2 = r2_score(actual, predicted)

    print(name)
    print(f"  MAE:  {mae:,.0f}")
    print(f"  MSE:  {mse:,.0f}")
    print(f"  RMSE: {rmse:,.0f}")
    print(f"  R2:   {r2:.3f}")


evaluate("Training set", y_train, model.predict(X_train))
evaluate("Test set", y_test, model.predict(X_test))
```

Expected output (approximately):

```
Training set
  MAE:  9,914
  MSE:  159,902,632
  RMSE: 12,645
  R2:   0.950

Test set
  MAE:  13,199
  MSE:  225,174,638
  RMSE: 15,006
  R2:   0.933
```

What each number means:

- **MAE ≈ 13,199:** on the test houses, predictions are off by about 13,199
  dollars on average, ignoring direction.
- **MSE ≈ 225,174,638:** the average squared error. The units are dollars
  squared, so the number is hard to read directly.
- **RMSE ≈ 15,006:** the typical error is in the same ballpark as 15,006
  dollars, with large errors counted more heavily.
- **R² ≈ 0.933:** the model explains about 93.3% of the variation in test
  prices. About 6.7% is still unexplained.

The test errors are higher than the training errors, which is normal: the model
was fitted to the training houses and is being judged on new ones. The gap is not
huge, so the line is not badly overfitting.

> Compare with the Day 04 baseline that always predicts the average price: its
> test RMSE was about **58,015**. Our line cuts that to about **15,006**, so it is
> clearly learning something useful.

## 9. Plot the Best-Fit Line

A picture makes the model easier to understand:

```python
grid = pd.DataFrame(
    {
        "area_sqft": np.linspace(
            df["area_sqft"].min(),
            df["area_sqft"].max(),
            100,
        )
    }
)

plt.figure(figsize=(8, 5))
plt.scatter(df["area_sqft"], df["price_usd"], alpha=0.4, label="Houses")
plt.plot(
    grid["area_sqft"],
    model.predict(grid),
    color="red",
    linewidth=2,
    label="Best-fit line",
)
plt.xlabel("Area (sqft)")
plt.ylabel("Price (USD)")
plt.title("Simple linear regression: area vs. price")
plt.legend()
plt.show()
```

What to look for:

- The red line follows the general upward trend.
- Many dots sit above or below the line. That is expected; the line cannot pass
  through every house.
- At the high end, the line may sit slightly below the dots, and in the middle it
  may sit slightly above them. That hints at the curved pattern.

## 10. Plot Actual vs. Predicted Values

A second useful plot compares the predictions with the true prices:

```python
test_predictions = model.predict(X_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, test_predictions, alpha=0.7)

low = min(y_test.min(), test_predictions.min())
high = max(y_test.max(), test_predictions.max())
plt.plot([low, high], [low, high], color="red", linestyle="--", label="Perfect")

plt.xlabel("Actual price (USD)")
plt.ylabel("Predicted price (USD)")
plt.title("Actual vs. predicted on the test set")
plt.legend()
plt.show()
```

How to read it:

- Points on the dashed line are perfect predictions.
- Points above the line are **underpredictions**.
- Points below the line are **overpredictions**.
- A good model forms a tight band around the dashed line. Ours is reasonably
  tight, but not perfect.

## 11. Plot the Residuals

Residual plots help us check whether the straight-line assumption is reasonable:

```python
plt.figure(figsize=(8, 5))
plt.scatter(test_predictions, y_test - test_predictions, alpha=0.7)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted price (USD)")
plt.ylabel("Residual (actual − predicted)")
plt.title("Residuals on the test set")
plt.show()
```

What to look for:

- We want a random cloud around the red zero line.
- A clear curve means the model is missing a curved pattern.
- A funnel shape (errors getting wider as predictions grow) means the error
  variance is not constant.

In this dataset, you may notice that residuals are often negative in the middle
and positive at the high end. The true relationship bends upward: very large
houses cost more than a straight line expects. Linear regression cannot bend by
itself. Day 06 shows how to add polynomial features to handle this.

## 12. Multiple Linear Regression

Now let us give the model more information: area and location. Location is text,
so we turn it into 0/1 columns with `pd.get_dummies`.

```python
multi_df = pd.get_dummies(
    df[["area_sqft", "location", "price_usd"]],
    columns=["location"],
    drop_first=True,
)

multi_df[["location_Suburb", "location_Town"]] = multi_df[
    ["location_Suburb", "location_Town"]
].astype(int)

multi_df.head()
```

Expected columns:

| area_sqft | price_usd | location_Suburb | location_Town |
|----------:|----------:|----------------:|--------------:|
| 814 | 34,000 | 1 | 0 |
| 2,458 | 173,000 | 1 | 0 |
| 2,171 | 148,500 | 0 | 1 |

`drop_first=True` leaves out one category, here `City`. That category becomes the
**baseline**. The other coefficients are read **relative to City**.

Now split and fit:

```python
X_multi = multi_df.drop(columns=["price_usd"])
y_multi = multi_df["price_usd"]

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi,
    y_multi,
    test_size=0.2,
    random_state=RANDOM_STATE,
)

multi_model = LinearRegression()
multi_model.fit(X_train_m, y_train_m)

print("Intercept:", round(multi_model.intercept_, 2))
print(pd.Series(multi_model.coef_, index=X_multi.columns).round(2))
```

Expected output (approximately):

```
Intercept: -34517.4
area_sqft          85.18
location_Suburb   -8480.62
location_Town    -10380.68
```

The learned rule is:

```
predicted_price = -34,517.40
                + 85.18 × area_sqft
                − 8,480.62 × location_Suburb
                − 10,380.68 × location_Town
```

How to interpret it:

- **Area:** for two houses in the same location, each extra square foot is
  associated with about 85 dollars more in predicted price.
- **Suburb:** for two houses with the same area, a Suburb house is predicted to
  cost about 8,481 dollars **less** than a City house.
- **Town:** for two houses with the same area, a Town house is predicted to cost
  about 10,381 dollars **less** than a City house.

Now evaluate it:

```python
evaluate("Multiple regression — training", y_train_m, multi_model.predict(X_train_m))
evaluate("Multiple regression — test", y_test_m, multi_model.predict(X_test_m))
```

Expected output (approximately):

```
Multiple regression — training
  MAE:  9,697
  MSE:  138,277,797
  RMSE: 11,759
  R2:   0.957

Multiple regression — test
  MAE:  11,912
  MSE:  193,188,581
  RMSE: 13,899
  R2:   0.943
```

## 13. Compare Simple and Multiple Regression

| Model | Features | Test MAE | Test RMSE | Test R² |
|-------|----------|---------:|----------:|--------:|
| Baseline (Day 04) | none, predict the mean | — | 58,015 | — |
| Simple regression | `area_sqft` | 13,199 | 15,006 | 0.933 |
| Multiple regression | `area_sqft` + `location` | 11,912 | 13,899 | 0.943 |

The multiple model is a little better: test RMSE drops from about 15,006 to
13,899, and R² rises from 0.933 to 0.943. Location adds some useful information,
but area still carries most of the signal.

This is a useful lesson: **more features do not always mean a much better
model.** Always compare against a simpler model and a baseline.

> In a real project, put the encoding and the model in a pipeline so the
> transformation is fitted on training data only. Day 04 showed that pattern, and
> Day 20 goes deeper.

## 14. Try It Yourself

Use the variables from this notebook.

1. Change the split to `test_size=0.3` and `random_state=0`. How do MAE, RMSE, and
   R² change?
2. Add `age_years` to the model. Six ages are missing, so impute them with the
   training median. Does the test RMSE improve?
3. Train a `DummyRegressor(strategy="mean")` baseline and compare its test RMSE
   with the simple regression.
4. Predict the price of a 3,500 sqft house. Is this interpolation or
   extrapolation? Would you trust the answer?
5. Remove houses with prices below 20,000 or above 230,000, then refit. Does the
   line change much?
6. Add `area_sqft ** 2` as a second feature. Does the model fit the curve better?
   (This is a preview of Day 06.)

Hints:

```python
X2 = df[["area_sqft", "age_years"]].copy()
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, test_size=0.2, random_state=RANDOM_STATE
)
X2_train = X2_train.copy()
X2_test = X2_test.copy()

median_age = X2_train["age_years"].median()
X2_train["age_years"] = X2_train["age_years"].fillna(median_age)
X2_test["age_years"] = X2_test["age_years"].fillna(median_age)

age_model = LinearRegression()
age_model.fit(X2_train, y2_train)
evaluate("Area + age — test", y2_test, age_model.predict(X2_test))
```

For question 6:

```python
curved = df[["area_sqft", "price_usd"]].copy()
curved["area_squared"] = curved["area_sqft"] ** 2

X_curved = curved[["area_sqft", "area_squared"]]
y_curved = curved["price_usd"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_curved, y_curved, test_size=0.2, random_state=RANDOM_STATE
)

curved_model = LinearRegression()
curved_model.fit(Xc_train, yc_train)
evaluate("Area + area² — test", yc_test, curved_model.predict(Xc_test))
```

Remember: a lower training error is not automatically better. Compare on the test
set.

## 15. What We Learned

In this practical we:

- Explored the relationship between house area and price.
- Split the data into training and test sets.
- Fitted a simple linear regression line.
- Read the slope and intercept in plain language.
- Made predictions for new houses.
- Inspected residuals to see where the model was too high or too low.
- Calculated MAE, MSE, RMSE, and R².
- Plotted the best-fit line, actual vs. predicted values, and residuals.
- Compared simple regression with multiple regression using area and location.
- Saw that the true relationship is slightly curved, which a straight line cannot
  fully capture.

## Outcome

Learners can train a simple and multiple linear regression model, interpret its
coefficients, make predictions, evaluate it with MAE, MSE, RMSE, and R², and use
plots and residuals to judge whether a straight line is appropriate.
