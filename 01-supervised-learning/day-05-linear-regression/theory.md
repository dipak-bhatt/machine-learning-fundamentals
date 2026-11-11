# Day 05 — Theory: Linear Regression from the Ground Up

**Module:** Module 2: Supervised Learning

Some machine learning questions ask for a **category**: Is this email spam or not?
Is this customer going to leave or stay? Other questions ask for a **number**:
How much will this house sell for? How many units will we sell next month? How
long will this delivery take?

**Linear regression** is the simplest tool for the second kind of question. It
learns a straight line that turns input values into a numerical prediction. In
this lesson we build that idea from scratch, with small examples before touching
a real dataset.

## In Plain English

**Think of it like this:** linear regression lays a ruler through the data and
asks, "what straight trend summarizes these points best?"

- **Regression** predicts a number; **classification** predicts a category.
- A line is `y = b + w*x`: `w` is the slope, `b` is the intercept.
- **Least squares** chooses the line whose squared errors are smallest.
- **R²** says how much variation the line explains; **RMSE/MAE** say how big a
  typical error is.
- A line only captures straight-line patterns — plot the residuals to catch
  curves.

## 1. Regression vs. Classification

Both regression and classification are **supervised learning**: we show the model
examples that include the correct answer, and it learns to predict that answer for
new examples.

| Question | Type | Target | Example prediction |
|----------|------|--------|--------------------|
| Is this email spam? | Classification | A category (`spam` / `not spam`) | `spam` |
| Will this student pass? | Classification | A category (`Yes` / `No`) | `Yes` |
| How much will this house sell for? | Regression | A number | `184,500` |
| How many minutes will delivery take? | Regression | A number | `27` |

The key difference is the **target**:

- A **classification** target is a label or category.
- A **regression** target is a number on a continuous scale.

Linear regression is for regression problems.

## 2. A Line Is a Rule

Before machine learning, imagine a very simple rule for taxi fares:

```
fare = 3 + 2 × kilometers
```

If you travel 0 km, the fare is 3. If you travel 5 km, the fare is
`3 + 2 × 5 = 13`. If you travel 10 km, the fare is 23.

| Kilometers | Calculation | Fare |
|-----------:|-------------|-----:|
| 0 | 3 + 2 × 0 | 3 |
| 5 | 3 + 2 × 5 | 13 |
| 10 | 3 + 2 × 10 | 23 |
| 15 | 3 + 2 × 15 | 33 |

This rule is a **line**. In mathematics we usually write it as:

```
y = m × x + b
```

- `x` is the input.
- `y` is the output.
- `m` is the **slope**: how much `y` changes when `x` increases by 1.
- `b` is the **intercept**: the value of `y` when `x = 0`.

In the taxi example, `m = 2` and `b = 3`. Each extra kilometer adds 2 to the
fare, and the starting fare is 3.

Linear regression does not receive the rule. It looks at examples of `x` and `y`
and **learns** the slope and intercept that fit those examples best.

## 3. Independent and Dependent Variables

In a dataset, each row is one example. Each column is either an input or the
answer we want to predict.

| Area (sqft) | Price (USD) |
|------------:|------------:|
| 800 | 42,000 |
| 1,200 | 75,000 |
| 1,600 | 108,000 |
| 2,000 | 141,000 |
| 2,400 | 174,000 |

- The **independent variable** (also called the **feature**, **input**, or `X`)
  is the information we are allowed to use. Here it is `area_sqft`.
- The **dependent variable** (also called the **target**, **label**, or `y`) is
  what we want to predict. Here it is `price_usd`.

The word "independent" does not mean the variables are unrelated. It means the
feature is not the thing we are trying to predict. The target **depends** on the
features, at least in theory.

In this tiny dataset, the pattern is perfectly straight: each extra 400 sqft adds
33,000. The rule is:

```
price = -24,000 + 82.5 × area
```

Real data is never this clean. There is noise, missing information, and
exceptions. Linear regression tries to find the line that is **closest** to all
the points, not a line that passes through every point.

## 4. Simple Linear Regression

**Simple linear regression** uses exactly one feature to predict the target:

```
y_hat = b0 + b1 × x
```

The little hat on `y_hat` means "predicted value". The true value is `y`.

- `b0` is the intercept.
- `b1` is the slope (also called the **coefficient**).
- `x` is the single feature.

With house area:

```
predicted_price = b0 + b1 × area_sqft
```

The model's job is to choose `b0` and `b1` so that the line is as close as
possible to the observed houses.

### The best-fit line

If we plot area on the horizontal axis and price on the vertical axis, each house
is a dot. Many straight lines could pass near the dots. The **best-fit line** is
the one that makes the overall distance from the dots to the line as small as
possible.

That sounds vague, so we need a precise way to measure "distance". That is where
**residuals** come in.

## 5. Residuals and Least Squares

A **residual** is the difference between the actual value and the predicted
value:

```
residual = actual − predicted
```

- A positive residual means the model predicted **too low**.
- A negative residual means the model predicted **too high**.
- A residual of zero means the prediction was exactly right.

### A tiny example

Suppose three houses have actual prices of 100,000, 150,000, and 200,000. A
candidate line predicts 110,000, 150,000, and 190,000.

| Actual | Predicted | Residual (actual − predicted) |
|-------:|----------:|------------------------------:|
| 100,000 | 110,000 | −10,000 |
| 150,000 | 150,000 | 0 |
| 200,000 | 190,000 | +10,000 |

If we simply added the residuals, the positive and negative values would cancel:
`−10,000 + 0 + 10,000 = 0`. That would make a bad line look perfect.

To avoid cancellation, we **square** each residual:

| Residual | Squared residual |
|---------:|-----------------:|
| −10,000 | 100,000,000 |
| 0 | 0 |
| +10,000 | 100,000,000 |

The **sum of squared errors** (SSE) is:

```
SSE = 100,000,000 + 0 + 100,000,000 = 200,000,000
```

Squaring does two useful things:

1. It makes every error positive, so errors cannot cancel.
2. It punishes large errors much more than small ones. An error of 20,000 is
   four times as costly as an error of 10,000, not twice.

The **least-squares** method chooses the line with the smallest SSE. "Least"
means smallest; "squares" refers to the squared errors.

### Comparing two candidate lines

Take three points: `(1, 2)`, `(2, 4)`, and `(3, 5)`.

Candidate A: `y = 1.5x + 0.5`

| x | Actual | Predicted | Residual | Squared |
|--:|-------:|----------:|---------:|--------:|
| 1 | 2 | 2.0 | 0 | 0 |
| 2 | 4 | 3.5 | +0.5 | 0.25 |
| 3 | 5 | 5.0 | 0 | 0 |

`SSE = 0.25`

Candidate B: `y = 2x − 0.5`

| x | Actual | Predicted | Residual | Squared |
|--:|-------:|----------:|---------:|--------:|
| 1 | 2 | 1.5 | +0.5 | 0.25 |
| 2 | 4 | 3.5 | +0.5 | 0.25 |
| 3 | 5 | 5.5 | −0.5 | 0.25 |

`SSE = 0.75`

Candidate A has the smaller SSE, so least squares prefers A. The true
least-squares line for these points is `y = 1.5x + 0.667`, which has an even
smaller SSE of about `0.167`. We do not need to calculate it by hand here; the
important idea is that the model searches for the line with the smallest total
squared error.

> Least squares is not magic. It is a clear rule: choose the line that minimizes
> the sum of squared residuals. Day 06 shows another way to find the same kind of
> answer using gradient descent.

## 6. Multiple Linear Regression

Most real questions have more than one useful input. A house price depends on
area, age, number of bedrooms, location, and more. **Multiple linear regression**
uses several features at once:

```
y_hat = b0 + b1×x1 + b2×x2 + b3×x3 + ... + bn×xn
```

For a house:

```
predicted_price = b0
                + b1 × area_sqft
                + b2 × age_years
                + b3 × bedrooms
```

Each feature gets its own coefficient. The model still learns a straight-line
relationship, but now it is a straight line in a higher-dimensional space. We
cannot draw it easily on a flat page, but the idea is the same: find the
coefficients that make the squared residuals as small as possible.

### What "holding other features constant" means

Suppose the model learns:

```
predicted_price = 20,000 + 80 × area + 5,000 × bedrooms
```

The coefficient `80` means:

> For two houses with the same number of bedrooms, each extra square foot is
> associated with about 80 dollars more in predicted price.

The coefficient `5,000` means:

> For two houses with the same area, each extra bedroom is associated with about
> 5,000 dollars more in predicted price.

This is why we say a coefficient is the effect of one feature **while holding the
other features constant**. The model is not saying that adding a bedroom
magically adds exactly 5,000 dollars. It is describing the pattern in the data.

## 7. Interpreting the Intercept and Coefficients

### The slope

The slope tells us how the prediction changes when a feature increases by one
unit. Units matter:

- If area is measured in square feet, the slope is **dollars per square foot**.
- If area is measured in square meters, the same relationship has a different
  slope number.

In today's practical, the simple model learns a slope of about `84.4`. That means:

> Each extra square foot is associated with about 84 dollars more in predicted
> price.

### The intercept

The intercept is the predicted value when every feature is zero. In the house
example, the intercept is about `−38,428`. That would be the predicted price of a
house with zero square feet.

A negative price for a zero-square-foot house is not meaningful. This is a
common situation: the intercept is needed to position the line, but it may not
describe a realistic case. Do not force a real-world meaning onto it.

### Categorical features

Some features are categories, not numbers. `location` might be `City`, `Suburb`,
or `Town`. Linear regression needs numbers, so we create **dummy variables**:
0/1 columns such as `location_Suburb` and `location_Town`.

One category is left out as the **baseline**. If `City` is the baseline, then:

- The `location_Suburb` coefficient compares Suburb houses with City houses of
  the same area.
- The `location_Town` coefficient compares Town houses with City houses of the
  same area.

In today's practical, the coefficients are about `−8,481` for Suburb and
`−10,381` for Town. Holding area constant, the model predicts that Suburb houses
sell for about 8,481 dollars less than City houses, and Town houses for about
10,381 dollars less.

## 8. Measuring How Good the Line Is

A model can fit the training data and still be wrong on new data. We always
evaluate on data the model has not seen. Four common regression metrics are MAE,
MSE, RMSE, and R².

### MAE: Mean Absolute Error

```
MAE = average of |actual − predicted|
```

MAE is the average size of the error, ignoring direction. It is in the same units
as the target, so a house-price MAE is measured in dollars. It is easy to explain:
"On average, our predictions are off by about 13,000 dollars."

### MSE: Mean Squared Error

```
MSE = average of (actual − predicted)²
```

MSE squares the errors before averaging, so it punishes large mistakes heavily.
Its units are squared (dollars squared for house prices), which makes it hard to
interpret directly. It is still useful for comparing models and for mathematical
convenience.

### RMSE: Root Mean Squared Error

```
RMSE = square root of MSE
```

RMSE brings the squared error back to the original units. Like MSE, it punishes
large errors more than MAE does. A house-price RMSE of 15,000 dollars means the
typical error is in the same ballpark as 15,000 dollars, with large errors
weighing more.

### R²: R-squared (the coefficient of determination)

R² compares the model with the simplest possible baseline: always predicting the
average target value.

```
R² = 1 − (sum of squared residuals) / (sum of squared deviations from the mean)
```

- `R² = 1` means perfect predictions.
- `R² = 0` means the model is no better than always predicting the mean.
- `R² < 0` means the model is worse than predicting the mean (possible on test
  data).

R² is unitless. An R² of `0.93` means the model explains about 93% of the
variation in the target, leaving about 7% unexplained. R² does not tell us whether
the errors are small enough for a real decision. That is why we look at MAE and
RMSE as well.

### A worked example

Actual prices: 100,000; 150,000; 200,000.
Predicted prices: 110,000; 150,000; 190,000.

Residuals: −10,000; 0; +10,000.

```
MAE  = (10,000 + 0 + 10,000) / 3 = 6,667
MSE  = (100,000,000 + 0 + 100,000,000) / 3 = 66,666,667
RMSE = sqrt(66,666,667) ≈ 8,165
```

The mean actual price is 150,000. The total squared deviation from the mean is:

```
(100,000 − 150,000)² + (150,000 − 150,000)² + (200,000 − 150,000)²
= 2,500,000,000 + 0 + 2,500,000,000
= 5,000,000,000
```

The sum of squared residuals is 200,000,000, so:

```
R² = 1 − 200,000,000 / 5,000,000,000 = 0.96
```

The model explains about 96% of the variation in these three examples, with an
average absolute error of about 6,667 dollars.

### Always compare with a baseline

A useful model should beat a trivial baseline. For regression, the simplest
baseline is:

```
predict the average of the training target for every row
```

If a complicated model cannot beat "just predict the average", it is not adding
value. In today's practical, the Day 04 baseline had a test RMSE of about 58,015
dollars. The linear model reduces that to about 15,006 dollars, so the line is
clearly learning something useful.

## 9. Assumptions of Linear Regression

Linear regression is simple and interpretable, but it works best when some
conditions hold. You do not need to memorize these on day one, but you should
recognize them.

| Assumption | Plain meaning | What to check |
|------------|---------------|---------------|
| Linearity | The average target changes roughly in a straight line with each feature | Scatter plots and residual plots |
| Independence | Rows do not influence each other | Think about how data was collected |
| Constant variance | The spread of residuals is similar across the range of predictions | Residual plot |
| Normal residuals | Residuals are roughly bell-shaped around zero | Histogram of residuals (mainly for confidence intervals) |
| No severe multicollinearity | Features are not almost duplicates of each other | Correlation matrix |
| No extreme outliers | A few points do not dominate the fit | Scatter plots and residual plots |

A **residual plot** shows residuals on the vertical axis and predictions (or a
feature) on the horizontal axis. We want a random cloud around zero. A curve or a
funnel shape suggests the model is missing something.

Assumptions are not a pass/fail exam. Real data often violates them a little.
The assumptions tell us when to be careful about interpretation, and when a
different model might be better.

## 10. Limitations of Linear Regression

Linear regression is a great first model, but it has real limits.

### It cannot bend by itself

A straight line cannot follow a curved pattern. If house prices grow faster for
very large houses, a single straight line will underpredict the largest houses
and overpredict some smaller ones. The practical shows this in the residual plot.

The fix is not to give up on regression. We can add **polynomial features**, such
as `area²`, so the model can bend. Day 06 covers that.

### It is sensitive to outliers

One unusual house sold at a strange price can pull the line toward it. Always
look for outliers and ask whether they are mistakes or genuine rare cases.

### It extrapolates badly

If the training data contains houses from 600 to 3,000 square feet, the model has
no evidence about a 10,000-square-foot mansion. The line will still produce a
number, but that number may be nonsense. Do not trust predictions far outside the
range of the training data.

### Correlation is not causation

A coefficient describes an association in the data, not proof that changing the
feature causes the target to change. Houses with more area cost more, but
"adding square feet" is not a simple experiment. There may be other reasons
behind the pattern.

### It only predicts numbers

If the target is a category such as `spam` or `not spam`, use a classification
model such as logistic regression (Day 07), not linear regression.

### It can miss important features

If a feature that matters is not in the data, the model cannot use it. A house
price model without location may be biased even if the line fits the training
data well.

## 11. Real-World Use Case: Delivery Time Prediction

A delivery company wants to estimate how long a delivery will take.

- **Row:** one delivery.
- **Features:** distance, number of stops, weather, time of day, driver
  experience.
- **Target:** delivery time in minutes.
- **Metric:** MAE, because "on average, we are off by 6 minutes" is easy for
  planners to understand. RMSE is also useful because late deliveries are costly.

A simple linear regression might learn:

```
predicted_minutes = 4 + 2.1 × distance_km + 1.5 × stops
```

Interpretation:

- For deliveries with the same number of stops, each extra kilometer is
  associated with about 2.1 more minutes.
- For deliveries of the same distance, each extra stop is associated with about
  1.5 more minutes.

What could go wrong:

- **Nonlinearity:** very long routes may have disproportionately long times.
- **Outliers:** a snowstorm creates a delivery that takes far longer than usual.
- **Missing features:** traffic and weather may matter a lot.
- **Extrapolation:** predicting a 500 km delivery when training data only went up
  to 50 km.
- **Leakage:** using "actual arrival time" as a feature would give away the
  answer.

The workflow is the same as Day 04: define the problem, split the data, fit
preprocessing on training data only, compare with a baseline, evaluate on unseen
data, and monitor after deployment.

## 12. Common Mistakes

1. Using linear regression for a yes/no target.
2. Evaluating the model only on training data.
3. Forgetting to split the data before fitting anything.
4. Reading coefficients as proof of causation.
5. Giving the intercept a real-world meaning when zero is impossible.
6. Ignoring the residual plot.
7. Extrapolating far beyond the training range.
8. Adding many highly correlated features and misreading their coefficients.
9. Letting a few extreme outliers control the line.
10. Comparing two models on different train/test splits.
11. Reporting R² alone without MAE or RMSE in the target's units.
12. Assuming a high R² means the model is useful for every decision.

## 13. Putting It Together

Before trusting a linear regression model, ask:

1. Is the target a number, not a category?
2. Did I split the data into training and test sets?
3. Did I fit preprocessing on the training set only?
4. Does the model beat the "predict the average" baseline?
5. Are the residuals randomly scattered around zero?
6. Do the coefficients make sense in the real world?
7. Am I interpreting association as association, not causation?
8. Am I predicting inside the range of the training data?
9. Are MAE and RMSE small enough for the actual decision?
10. Does R² tell the same story as the error metrics?

If the answers are reasonable, you have a solid first regression model.

## Outcome

Learners can explain the difference between regression and classification,
describe a line using a slope and intercept, explain how least squares chooses a
best-fit line, interpret simple and multiple regression coefficients, evaluate a
regression model with MAE, MSE, RMSE, and R², and recognize the main assumptions
and limitations of linear regression.
