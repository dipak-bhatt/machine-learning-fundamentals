# House Price Prediction

Predict median house values for California districts with regression. This is
the end-to-end regression project for Module 2 (supervised learning).

## Dataset

`sklearn.datasets.fetch_california_housing` ships with scikit-learn and is
downloaded once into the scikit-learn cache.

- 20,640 rows, 8 numeric features
- Target: median house value in units of 100k USD

## What `train.py` does

1. Loads the dataset and prints shape, missing values, summary statistics, and
   feature-target correlations.
2. Saves exploratory plots:
   - target distribution
   - correlation heatmap
3. Splits the data into 80% train and 20% test.
4. Trains and compares:
   - `LinearRegression` baseline
   - `Ridge(alpha=1.0)` inside a `StandardScaler` pipeline
5. Evaluates MAE, RMSE, and R-squared on the held-out test set.
6. Saves predicted-vs-actual and residual plots for the best model.
7. Saves the best model with joblib.

## Run

```bash
python projects/house-price-prediction/train.py
```

## Outputs

- `outputs/target_distribution.png`
- `outputs/correlation_heatmap.png`
- `outputs/model_diagnostics.png`
- `outputs/metrics.json`
- `models/house_price_model.joblib`

The baseline and Ridge models usually score similarly. Ridge helps when
features are correlated and adds regularization; check `metrics.json` to see
which model wins on this split.

## Ideas to extend

- Tune `alpha` with `RidgeCV` or `GridSearchCV`.
- Add engineered features such as rooms per household.
- Compare against `RandomForestRegressor` or `HistGradientBoostingRegressor`.
- Map residuals geographically using latitude and longitude.
