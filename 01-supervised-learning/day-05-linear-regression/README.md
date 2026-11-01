# Day 05 — Linear Regression: Predicting Numerical Values

**Module:** Module 2: Supervised Learning

Some questions have a number as the answer: How much will this house sell for?
How many customers will visit next week? How long will a delivery take?
**Linear regression** is the simplest way to answer them. It learns a straight
line that turns input values into a numerical prediction.

Today we start with the very basics: what a line is, what a slope and intercept
mean, how the best-fit line is chosen, and how to measure whether its predictions
are any good. Then we train our first regression model on real-looking house
data and interpret what it learned.

See the detailed content in:

- [Theory](theory.md) — regression vs. classification, independent and dependent
  variables, the best-fit line, residuals and least squares, simple and multiple
  linear regression, coefficients and intercept, MAE/MSE/RMSE/R², assumptions,
  limitations, and common mistakes.
- [Practical](practical.md) — load the Day 04 house-price dataset, fit a simple
  regression line, make predictions, inspect residuals, calculate four evaluation
  metrics, plot the fit, and compare simple and multiple regression.

## Files

- `../../00-foundations/day-04-bias-variance-ml-workflow/house_prices.csv` — the dataset reused from
  Day 04. It contains 140 synthetic houses with `area_sqft`, `age_years`,
  `location`, and `price_usd`. Reusing it lets us focus on the new model instead
  of learning another dataset.

The notebook in the practical must be created inside this Day 05 folder so the
relative path to the CSV works.

## Outcome

Learners can explain what linear regression does, read the slope and intercept of
a fitted line, make and interpret predictions, evaluate a regression model with
MAE, MSE, RMSE, and R², and recognize when a straight line is not a good fit.
