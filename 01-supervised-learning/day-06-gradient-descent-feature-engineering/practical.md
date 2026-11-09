# Day 06 — Practical: Gradient Descent and Feature Engineering

**Module:** Module 2: Supervised Learning

In this practical we move from the formula for a gradient to a working model. We
will:

- Perform one gradient descent update by hand.
- Implement batch gradient descent in NumPy and watch the loss fall.
- See how the learning rate changes convergence.
- Reuse the Day 04 house data to compare plain and scaled regression.
- Add polynomial features to capture a curve.
- Compare ordinary, Ridge, and Lasso regression on the same held-out rows.

The notebook should be created inside this Day 06 folder. The CSV path below
assumes that, the same convention used in Day 05.

| Column | Meaning | Type |
|--------|---------|------|
| `area_sqft` | Size of the house in square feet | Number |
| `age_years` | Age of the house in years | Number (6 missing) |
| `location` | `City`, `Suburb`, or `Town` | Category |
| `price_usd` | Selling price in US dollars | Target |

## 1. Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
np.set_printoptions(precision=3, suppress=True)
```

`RANDOM_STATE` keeps the split repeatable. With the same seed, you and I get the
same rows in training and test.

## 2. Warm-Up: One Update by Hand

We have one example: input `x = 2`, actual answer `y = 5`. Start with intercept
`b = 0` and slope `w = 0`, and use learning rate `alpha = 0.1`.

**Step 1: Predict.**

```text
prediction = b + w * x = 0 + 0 * 2 = 0
```

**Step 2: Find the error.**

```text
error = prediction - actual = 0 - 5 = -5
```

**Step 3: Compute both gradients.** There is one row, so `n = 1`:

```text
gradient_b = (2/n) * sum(error)     = 2 * (-5) = -10
gradient_w = (2/n) * sum(error * x) = 2 * (-5) * 2 = -20
```

**Step 4: Update both parameters.**

```text
new b = 0 - 0.1 * (-10) = 1
new w = 0 - 0.1 * (-20) = 2
```

**Step 5: Check the new prediction.**

```text
new prediction = 1 + 2 * 2 = 5
```

The prediction now matches this single example. Let us confirm with code:

```python
x, y = 2, 5
b, w, alpha = 0.0, 0.0, 0.1

prediction = b + w * x
error = prediction - y
gradient_b = 2 * error
gradient_w = 2 * error * x

b = b - alpha * gradient_b
w = w - alpha * gradient_w

print("prediction:", prediction)
print("error:", error)
print("gradients:", gradient_b, gradient_w)
print("new b, w:", b, w)
print("new prediction:", b + w * x)
```

Expected output:

```text
prediction: 0.0
error: -5.0
gradients: -10.0 -20.0
new b, w: 1.0 2.0
new prediction: 5.0
```

One example is not enough to learn from in practice: the update fits this row
perfectly and knows nothing about the others. Real training sets have many rows,
so we average the errors, as the next section does.

> The theory file contains a two-row version of this calculation with several
> passes. That table is worth re-reading if the signs feel confusing.

## 3. Implement Batch Gradient Descent in NumPy

We will create data from a known rule, `y = 3 + 2x`, with a little noise. That
lets us check whether the algorithm recovers an intercept near 3 and a slope near
2.

```python
rng = np.random.default_rng(RANDOM_STATE)

X_toy = np.linspace(0, 10, 60).reshape(-1, 1)
y_toy = 3 + 2 * X_toy[:, 0] + rng.normal(0, 1, size=len(X_toy))

X_toy_train, X_toy_test, y_toy_train, y_toy_test = train_test_split(
    X_toy, y_toy, test_size=0.2, random_state=RANDOM_STATE
)
```

The function below standardizes the input using training data only. It then adds
a column of ones so the intercept is updated together with the slope.

```python
def fit_gradient_descent(X, y, learning_rate=0.1, iterations=1000):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # First column is for the intercept; remaining columns are features.
    X_with_intercept = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    weights = np.zeros(X_with_intercept.shape[1])
    losses = []

    for _ in range(iterations):
        predictions = X_with_intercept @ weights
        errors = predictions - y
        gradient = (2 / len(y)) * (X_with_intercept.T @ errors)
        weights -= learning_rate * gradient
        losses.append(np.mean(errors ** 2))

    return scaler, weights, losses


scaler, weights, losses = fit_gradient_descent(
    X_toy_train, y_toy_train, learning_rate=0.1, iterations=1000
)

print("Intercept in standardized-input model:", round(weights[0], 2))
print("Weight for standardized x:", round(weights[1], 2))
print("Final training MSE:", round(losses[-1], 3))
```

Expected output:

```text
Intercept in standardized-input model: 12.71
Weight for standardized x: 5.62
Final training MSE: 0.606
```

The intercept is near the average `y` in the training set. Because the input was
standardized, the weight is the change in prediction for a **one
standard-deviation** increase in `x`, not for one raw unit of `x`. We can convert
back to the original units:

```python
w_raw = weights[1] / scaler.scale_[0]
b_raw = weights[0] - w_raw * scaler.mean_[0]

print("Raw-scale intercept and slope:", round(b_raw, 2), round(w_raw, 2))
```

Expected output:

```text
Raw-scale intercept and slope: 3.04 2.0
```

That is very close to the true rule `y = 3 + 2x`. This is a useful pattern: fit
on scaled features for stable optimization, then convert coefficients if you need
to explain them in the original units.

Note that `weights` has two entries. The first is the intercept, and the second is
the slope, because we built the matrix with a leading column of ones.

## 4. Plot the Loss and Tune the Learning Rate

A loss curve is the quickest way to see whether optimization is working. The loss
should fall quickly at first and then flatten.

```python
plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.title("Gradient descent loss")
plt.show()
```

What to look for:

- A steep drop near the start means the parameters were far from the bottom.
- A long flat tail means the updates are barely changing anything.
- A rising or spiky curve means the learning rate is too large.

Now compare several learning rates:

```python
for lr in [0.001, 0.01, 0.1, 0.5, 1.0]:
    _, _, lr_losses = fit_gradient_descent(
        X_toy_train, y_toy_train, learning_rate=lr, iterations=1000
    )
    print(
        f"learning rate {lr:<5}"
        f"  first MSE {lr_losses[0]:7.2f}"
        f"  final MSE {lr_losses[-1]:7.2f}"
    )
```

Expected output:

```text
learning rate 0.001  first MSE  193.89  final MSE    4.15
learning rate 0.01   first MSE  193.89  final MSE    0.61
learning rate 0.1    first MSE  193.89  final MSE    0.61
learning rate 0.5    first MSE  193.89  final MSE    0.61
learning rate 1.0    first MSE  193.89  final MSE  193.89
```

How to read the table:

- `0.001` is too small: after 1,000 updates it has not reached the bottom.
- `0.01` converges, but more slowly than the larger values.
- `0.1` and `0.5` converge quickly to the same loss.
- `1.0` makes no progress at all; the parameters flip back and forth.

Why does `1.0` fail? After standardization the two columns have similar scale, so
for this toy problem the safe range is roughly `alpha < 1`. Exactly `1.0` sits on
the edge and oscillates. With real, correlated features the safe upper limit is
usually smaller, which is one reason scaling matters.

If a run diverges, stop it and use a smaller learning rate. A very large rate can
produce `inf` or `nan`, and once that happens every later number is meaningless.

## 5. Load and Inspect the House Data

```python
df = pd.read_csv(
    "../../00-foundations/day-04-bias-variance-ml-workflow/house_prices.csv"
)
df.head()
```

If you get a `FileNotFoundError`, your notebook is not inside the Day 06 folder.
The path above goes up two folders to the repository root, then into the Day 04
folder.

Check the shape, missing values, and summary statistics:

```python
print("Shape:", df.shape)
print(df.isnull().sum())
df.describe().round(1)
```

Expected output (approximately):

```text
Shape: (140, 4)
area_sqft    0
age_years    6
location     0
price_usd    0
dtype: int64
```

| Statistic | `area_sqft` | `age_years` | `price_usd` |
|-----------|------------:|------------:|------------:|
| count | 140.0 | 134.0 | 140.0 |
| mean | 1,841.2 | 19.2 | 116,428.6 |
| std | 643.0 | 11.6 | 56,945.8 |
| min | 617.0 | 0.0 | 16,000.0 |
| 25% | 1,344.0 | 10.0 | 67,875.0 |
| 50% | 1,910.5 | 18.0 | 110,250.0 |
| 75% | 2,424.8 | 28.8 | 161,000.0 |
| max | 2,942.0 | 40.0 | 238,000.0 |

The dataset has 140 houses. Only `age_years` has missing values, and we will not
use age in the first models, so we can use all 140 rows. Prices run from about
16,000 to 238,000 dollars.

## 6. Look at the Relationship

Before fitting, look at the data. A scatter plot shows whether a straight line is
a reasonable starting point.

```python
plt.figure(figsize=(8, 5))
plt.scatter(df["area_sqft"], df["price_usd"], alpha=0.5)
plt.xlabel("Area (sqft)")
plt.ylabel("Price (USD)")
plt.title("House area vs. price")
plt.show()
```

What to look for:

- The points rise from left to right, so area and price are positively associated.
- The cloud is not perfectly straight. Price seems to grow faster for the largest
  houses.
- A few houses sit far above or below the main cloud. They may be unusual cases
  or simply noise.

A straight line can still be a useful first model, even when the pattern bends.
The polynomial model in Section 10 will test whether the curve is real.

## 7. Split Once and Reuse the Same Split

Keep the same test rows for every model so the comparison is fair.

```python
X = df[["area_sqft"]]
y = df["price_usd"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
```

Expected output:

```text
Training rows: 112
Test rows: 28
```

Two details matter:

- `X = df[["area_sqft"]]` keeps `X` as a 2D table with one column. Writing
  `df["area_sqft"]` gives a 1D Series, and scikit-learn will complain.
- We split **before** fitting any scaler or model, so the test houses stay unseen.

We will reuse one small reporting helper so every model is measured the same way:

```python
def report(name, model):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    print(name)
    print(f"  Train RMSE: {mean_squared_error(y_train, train_pred) ** 0.5:,.0f}")
    print(f"  Test RMSE:  {mean_squared_error(y_test, test_pred) ** 0.5:,.0f}")
    print(f"  Test MAE:   {mean_absolute_error(y_test, test_pred):,.0f}")
    print(f"  Test R2:    {r2_score(y_test, test_pred):.3f}")
```

## 8. Establish a Baseline

The simplest regression baseline predicts the training mean for every row. If a
real model cannot beat it, the model is not adding value.

```python
baseline_model = DummyRegressor(strategy="mean").fit(X_train, y_train)
report("Baseline (predict the mean)", baseline_model)
```

Expected output:

```text
Baseline (predict the mean)
  Train RMSE: 56,420
  Test RMSE:  58,015
  Test MAE:   50,997
  Test R2:    -0.000
```

An R² of about zero is exactly what we expect: the mean baseline explains none of
the variation. Keep this test RMSE of about **58,015** in mind for comparison.

## 9. Ordinary Regression and Feature Scaling

For ordinary least squares, scaling should not change the predictions; it only
changes the units of the coefficients. Scaling becomes important for gradient
descent and for regularized models.

```python
plain_model = LinearRegression().fit(X_train, y_train)
scaled_model = make_pipeline(StandardScaler(), LinearRegression()).fit(
    X_train, y_train
)

print("Slope:", round(plain_model.coef_[0], 2))
print("Intercept:", round(plain_model.intercept_, 2))
report("Plain linear regression", plain_model)
report("Scaled linear regression", scaled_model)
```

Expected output:

```text
Slope: 84.4
Intercept: -38428.23
Plain linear regression
  Train RMSE: 12,645
  Test RMSE:  15,006
  Test MAE:   13,199
  Test R2:    0.933
Scaled linear regression
  Train RMSE: 12,645
  Test RMSE:  15,006
  Test MAE:   13,199
  Test R2:    0.933
```

The metrics are identical, as predicted. Compare the test RMSE of 15,006 with the
baseline's 58,015: the line is clearly learning something useful.

In plain units, the model learned:

```text
predicted_price = -38,428.23 + 84.40 * area_sqft
```

Each extra square foot is associated with about 84 dollars more in predicted
price. The negative intercept positions the line; it is not a realistic price for
a zero-square-foot house.

Scaling does not improve this model's predictions. It improves the **optimization
process** and makes penalties fair. Keep that distinction clear.

## 10. Add a Polynomial Feature

`PolynomialFeatures` adds `area_sqft^2`. The pipeline ensures the expansion and
the scaling are fitted consistently as part of the model.

```python
linear_model = make_pipeline(
    StandardScaler(),
    LinearRegression(),
).fit(X_train, y_train)

quadratic_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    LinearRegression(),
).fit(X_train, y_train)

report("Straight-line model", linear_model)
report("Quadratic model", quadratic_model)
```

Expected output:

```text
Straight-line model
  Train RMSE: 12,645
  Test RMSE:  15,006
  Test MAE:   13,199
  Test R2:    0.933
Quadratic model
  Train RMSE: 9,030
  Test RMSE:  9,991
  Test MAE:   7,866
  Test R2:    0.970
```

The quadratic model lowers test RMSE from 15,006 to 9,991 and raises R² from 0.933
to 0.970. The curve feature is helping on this split.

Plot both fitted curves:

```python
grid = pd.DataFrame(
    {"area_sqft": np.linspace(df["area_sqft"].min(), df["area_sqft"].max(), 200)}
)

plt.figure(figsize=(8, 5))
plt.scatter(X["area_sqft"], y, alpha=0.35, label="Houses")
plt.plot(grid["area_sqft"], linear_model.predict(grid), label="Straight line")
plt.plot(grid["area_sqft"], quadratic_model.predict(grid), label="Quadratic")
plt.xlabel("Area (sqft)")
plt.ylabel("Price (USD)")
plt.title("Linear and quadratic models")
plt.legend()
plt.show()
```

The quadratic curve should sit closer to the dots at the high end of the area
range, where the straight line falls below the pattern.

### Check whether more degrees help

Before choosing a model, compare several polynomial degrees on the same split:

```python
for degree in [1, 2, 3, 5]:
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        LinearRegression(),
    ).fit(X_train, y_train)

    train_rmse = mean_squared_error(y_train, model.predict(X_train)) ** 0.5
    test_rmse = mean_squared_error(y_test, model.predict(X_test)) ** 0.5
    print(f"degree {degree}: train RMSE {train_rmse:,.0f}  test RMSE {test_rmse:,.0f}")
```

Expected output:

```text
degree 1: train RMSE 12,645  test RMSE 15,006
degree 2: train RMSE 9,030  test RMSE 9,991
degree 3: train RMSE 9,007  test RMSE 10,144
degree 5: train RMSE 8,927  test RMSE 10,200
```

This is the overfitting lesson in one table. Degree 2 is best on test. Degrees 3
and 5 keep improving the **training** error while making the **test** error worse.
The extra wiggles are fitting noise, not signal.

## 11. Compare Ordinary Regression and Ridge

Ridge adds a penalty for large coefficients. We standardize the generated
polynomial features so the penalty treats them fairly.

```python
ridge_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    Ridge(alpha=10.0),
).fit(X_train, y_train)

report("Quadratic ordinary regression", quadratic_model)
report("Quadratic Ridge (alpha=10)", ridge_model)
```

Expected output:

```text
Quadratic ordinary regression
  Train RMSE: 9,030
  Test RMSE:  9,991
  Test MAE:   7,866
  Test R2:    0.970
Quadratic Ridge (alpha=10)
  Train RMSE: 10,064
  Test RMSE:  12,262
  Test MAE:   10,804
  Test R2:    0.955
```

Here the penalty is too strong: Ridge makes both training and test errors worse.
That is not a failure of Ridge, it is a sign that `alpha=10` is too large for this
small dataset. Explore the penalty strength:

```python
for alpha_ridge in [0.01, 0.1, 1, 10, 100, 1000]:
    ridge = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        Ridge(alpha=alpha_ridge),
    ).fit(X_train, y_train)

    train_rmse = mean_squared_error(y_train, ridge.predict(X_train)) ** 0.5
    test_rmse = mean_squared_error(y_test, ridge.predict(X_test)) ** 0.5
    print(f"alpha {alpha_ridge:<6}: train RMSE {train_rmse:,.0f}  test RMSE {test_rmse:,.0f}")
```

Expected output:

```text
alpha 0.01  : train RMSE 9,030  test RMSE 9,997
alpha 0.1   : train RMSE 9,034  test RMSE 10,048
alpha 1     : train RMSE 9,196  test RMSE 10,517
alpha 10    : train RMSE 10,064  test RMSE 12,262
alpha 100   : train RMSE 19,900  test RMSE 23,075
alpha 1000  : train RMSE 46,509  test RMSE 48,558
```

As `alpha` grows, the model becomes less flexible:

- Small penalties behave almost like ordinary regression.
- Medium penalties shrink the coefficients a little.
- Large penalties shrink them so much that the model underfits, and eventually it
  is barely better than a flat line.

The best value depends on the data. In a real project, choose it with
cross-validation on the **training** data, then use the test set once for a final
check. Selecting `alpha` by repeatedly comparing test RMSE is a subtle form of
tuning on the test set, and it makes the final score too optimistic.

## 12. Try Lasso

Lasso uses an absolute-value penalty and can set weights to zero. It also needs
scaled features. Use the same polynomial pipeline:

```python
lasso_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    Lasso(alpha=0.1, max_iter=10000),
).fit(X_train, y_train)

report("Quadratic Lasso (alpha=0.1)", lasso_model)
```

Expected output:

```text
Quadratic Lasso (alpha=0.1)
  Train RMSE: 9,030
  Test RMSE:  9,991
  Test MAE:   7,866
  Test R2:    0.970
```

The result matches ordinary quadratic regression because `alpha=0.1` is too weak
to remove either feature. With only two generated features this is mainly a
demonstration. To see the zeroing behavior, add an exact duplicate of the feature
and compare Ridge with Lasso:

```python
X_dup = df[["area_sqft"]].copy()
X_dup["area_copy"] = X_dup["area_sqft"]

X_dup_train, X_dup_test, y_dup_train, y_dup_test = train_test_split(
    X_dup, df["price_usd"], test_size=0.2, random_state=RANDOM_STATE
)

ridge_dup = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(
    X_dup_train, y_dup_train
)
lasso_dup = make_pipeline(StandardScaler(), Lasso(alpha=1000, max_iter=100000)).fit(
    X_dup_train, y_dup_train
)

print("Ridge weights:", np.round(ridge_dup.named_steps["ridge"].coef_, 0))
print("Lasso weights:", np.round(lasso_dup.named_steps["lasso"].coef_, 0))
```

Expected output:

```text
Ridge weights: [26317. 26317.]
Lasso weights: [53985.     0.]
```

Ridge splits the weight evenly between the two identical features. Lasso keeps one
and sets the other to zero. Neither is wrong; they make different tradeoffs.

Because the coefficients are on **standardized** features, their raw magnitudes
are not directly in dollars per square foot. To interpret them, convert back to
the original units or fit a simpler unscaled model for explanation.

## 13. Compare All Models

All rows below use the same 28 test houses:

| Model | Test MAE | Test RMSE | Test R² |
|-------|---------:|----------:|--------:|
| Baseline (predict the mean) | 50,997 | 58,015 | 0.000 |
| Linear regression | 13,199 | 15,006 | 0.933 |
| Quadratic regression | 7,866 | 9,991 | 0.970 |
| Quadratic Ridge (`alpha=10`) | 10,804 | 12,262 | 0.955 |
| Quadratic Lasso (`alpha=0.1`) | 7,866 | 9,991 | 0.970 |

What the table tells us:

- Every real model beats the baseline by a wide margin.
- The quadratic term captures a real curve in this dataset.
- Ridge with `alpha=10` is worse than the unpenalized quadratic model here.
- Lasso with `alpha=0.1` is effectively unregularized for these two features.

The lesson is not "Ridge is bad" or "Lasso is useless". The lesson is that
**penalty strength is a hyperparameter**. It must be chosen on data the model was
not trained on, but also not the final test set. Cross-validation is the standard
tool for that, and Day 19 to Day 20 cover it.

## 14. Try It Yourself

Use the variables from this notebook.

1. Run gradient descent for 20, 100, and 1,000 iterations. Record the final
   training MSE each time. How many iterations does this problem really need?
2. Try learning rates of `0.005`, `0.05`, and `0.2`. Which converge smoothly?
   Which are too slow? Which become unstable?
3. Change the polynomial degree from 2 to 3 and 5. Compare train and test RMSE.
   Does the training score keep improving while the test score gets worse?
4. Try Ridge `alpha` values of `0.01`, `1`, `10`, and `100`. Which has the best
   test RMSE? Why should you avoid selecting the final model from this test table
   in a larger project?
5. Add `age_years` to the model. It has missing values. Impute them using the
   training median only, then compare test RMSE. Put imputation and scaling inside
   a pipeline so the test rows never influence the fitted statistics.
6. Create a feature that is an exact copy of `area_sqft`. Compare Ridge and Lasso
   coefficients as in Section 12. Describe the difference in plain words.
7. Explain in your own words why standardization helps gradient descent and
   Ridge, but does not change the predictions of ordinary least squares.
8. Plot the Ridge test RMSE against `alpha`. Where does the curve bottom out?
   What happens at the two extremes?

Hints for question 5:

```python
from sklearn.impute import SimpleImputer

X_age = df[["area_sqft", "age_years"]]
y_age = df["price_usd"]

X_age_train, X_age_test, y_age_train, y_age_test = train_test_split(
    X_age, y_age, test_size=0.2, random_state=RANDOM_STATE
)

age_model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LinearRegression(),
).fit(X_age_train, y_age_train)

age_pred = age_model.predict(X_age_test)
print("Test RMSE:", round(mean_squared_error(y_age_test, age_pred) ** 0.5))
```

The imputer learns the median from `X_age_train` only. When the pipeline later
transforms the test rows, it reuses that training median. That is the safe
pattern from Day 04.

## 15. What We Learned

- A gradient is the direction of fastest increase in loss; subtracting a scaled
  gradient moves parameters toward lower loss.
- Batch gradient descent repeats: predict, compute errors, compute gradients,
  update.
- The learning rate and feature scale determine whether updates are useful.
- Standardization helps optimization and regularization; for ordinary least
  squares it leaves predictions unchanged.
- Polynomial features let a linear model represent curves, but higher degrees can
  overfit.
- Ridge shrinks weights; Lasso can shrink some weights to exactly zero.
- Penalty strength is a hyperparameter that belongs to validation, not to the
  final test set.
- We compared every model on the same unseen rows and against a baseline.

## Outcome

Learners can implement a basic batch gradient descent loop, read its loss curve,
tune a learning rate, explain why features are scaled, create polynomial features,
and compare straight-line, polynomial, Ridge, and Lasso regression using honest
held-out evaluation.
