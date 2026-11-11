"""House price prediction with linear regression and Ridge regression.

Run from the repository root:
    python projects/house-price-prediction/train.py
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"


def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame.rename(columns={"MedHouseVal": "MedHouseValue"})
    return df


def explore_data(df):
    print("Shape:", df.shape)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nSummary statistics:")
    print(df.describe().round(2))
    print("\nCorrelation with the target:")
    print(
        df.corr(numeric_only=True)["MedHouseValue"]
        .drop("MedHouseValue")
        .sort_values(ascending=False)
        .round(3)
    )


def save_target_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["MedHouseValue"], bins=50, color="#4c72b0", edgecolor="white")
    ax.set_xlabel("Median house value (100k USD)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of California house prices")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "target_distribution.png", dpi=150)
    plt.close(fig)


def save_correlation_heatmap(df):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "model": name,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }
    return metrics, model, predictions


def save_diagnostics(name, y_test, predictions):
    residuals = y_test - predictions

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(predictions, y_test, s=8, alpha=0.4)
    limits = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    axes[0].plot(limits, limits, color="red", linewidth=1.5, label="Perfect prediction")
    axes[0].set_xlabel("Predicted value")
    axes[0].set_ylabel("Actual value")
    axes[0].set_title(f"{name}: predicted vs actual")
    axes[0].legend()

    axes[1].scatter(predictions, residuals, s=8, alpha=0.4)
    axes[1].axhline(0, color="red", linewidth=1.5)
    axes[1].set_xlabel("Predicted value")
    axes[1].set_ylabel("Residual")
    axes[1].set_title(f"{name}: residuals")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_diagnostics.png", dpi=150)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    explore_data(df)
    save_target_distribution(df)
    save_correlation_heatmap(df)

    X = df.drop(columns="MedHouseValue")
    y = df["MedHouseValue"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    }

    results = []
    fitted_models = {}
    predictions = {}
    for name, model in candidates.items():
        metrics, fitted, preds = evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )
        results.append(metrics)
        fitted_models[name] = fitted
        predictions[name] = preds

    results_df = pd.DataFrame(results).sort_values("rmse")
    print("\nModel comparison (lower MAE/RMSE and higher R2 are better):")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    best_predictions = predictions[best_name]
    print(f"\nBest model: {best_name}")

    save_diagnostics(best_name, y_test, best_predictions)

    model_path = MODEL_DIR / "house_price_model.joblib"
    joblib.dump(best_model, model_path)
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nSaved best model to {model_path}")
    print(f"Saved plots and metrics to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
