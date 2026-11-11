"""Customer churn prediction with preprocessing pipelines and model comparison.

Run from the repository root:
    python projects/customer-churn-prediction/train.py

The Telco churn dataset is downloaded once and cached in data/raw/. If the
download is unavailable, a deterministic synthetic telco-style dataset is used.
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data" / "raw"


def make_synthetic_churn(n_samples=2000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n_samples)
    monthly = np.round(rng.uniform(20, 120, n_samples), 2)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n_samples, p=[0.55, 0.25, 0.20]
    )
    internet = rng.choice(
        ["DSL", "Fiber optic", "No"], n_samples, p=[0.35, 0.45, 0.20]
    )
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n_samples,
        p=[0.35, 0.20, 0.25, 0.20],
    )
    tech_support = rng.choice(
        ["Yes", "No", "No internet service"], n_samples, p=[0.30, 0.50, 0.20]
    )
    paperless = rng.choice(["Yes", "No"], n_samples, p=[0.60, 0.40])
    senior = rng.integers(0, 2, n_samples)
    partner = rng.choice(["Yes", "No"], n_samples, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n_samples, p=[0.30, 0.70])

    logit = (
        -1.2
        - 0.035 * (tenure - 30)
        + 0.012 * (monthly - 70)
        + np.where(contract == "Month-to-month", 1.0, 0.0)
        + np.where(contract == "Two year", -0.9, 0.0)
        + np.where(internet == "Fiber optic", 0.45, 0.0)
        + np.where(payment == "Electronic check", 0.35, 0.0)
        + np.where(tech_support == "Yes", -0.35, 0.0)
        + 0.30 * senior
    )
    churn_probability = 1.0 / (1.0 + np.exp(-logit))
    churn = rng.binomial(1, churn_probability)

    total_charges = np.round(
        tenure * monthly * rng.normal(1.0, 0.01, n_samples), 2
    )

    return pd.DataFrame(
        {
            "customerID": [f"SYN-{index:05d}" for index in range(n_samples)],
            "gender": rng.choice(["Female", "Male"], n_samples),
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], n_samples, p=[0.90, 0.10]),
            "InternetService": internet,
            "TechSupport": tech_support,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total_charges,
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )


def load_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / "telco_churn.csv"
    if cache_path.exists():
        print(f"Loading cached dataset from {cache_path}")
        return pd.read_csv(cache_path)

    try:
        df = pd.read_csv(TELCO_URL)
        df.to_csv(cache_path, index=False)
        print(f"Downloaded dataset and cached it at {cache_path}")
        return df
    except Exception as exc:  # network failures should not block the lesson
        print(f"Could not download the Telco dataset ({exc}).")
        print("Falling back to a deterministic synthetic telco-style dataset.")
        return make_synthetic_churn()


def clean_data(df):
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns="customerID")
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0})
    df = df.dropna(subset=["Churn"])
    df["Churn"] = df["Churn"].astype(int)
    return df.reset_index(drop=True)


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )


def save_exploration_plots(df, churn_rate):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Churn"].map({0: "No churn", 1: "Churn"}).value_counts()
    bars = ax.bar(counts.index, counts.values, color=["#4c72b0", "#c44e52"])
    ax.bar_label(bars)
    ax.set_ylabel("Customers")
    ax.set_title(f"Class balance (churn rate = {churn_rate:.1%})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_balance.png", dpi=150)
    plt.close(fig)

    if "Contract" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        rates = df.groupby("Contract")["Churn"].mean().sort_values()
        bars = ax.bar(rates.index, rates.values, color="#55a868")
        ax.bar_label(bars, fmt="%.1f%%", labels=[f"{v:.1%}" for v in rates.values])
        ax.set_ylabel("Churn rate")
        ax.set_title("Churn rate by contract type")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "churn_by_contract.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, name in [(0, "No churn"), (1, "Churn")]:
        subset = df[df["Churn"] == label]
        axes[0].hist(subset["tenure"], bins=25, alpha=0.6, label=name)
        axes[1].hist(subset["MonthlyCharges"], bins=25, alpha=0.6, label=name)
    axes[0].set_xlabel("Tenure (months)")
    axes[0].set_ylabel("Customers")
    axes[0].set_title("Tenure by churn status")
    axes[1].set_xlabel("Monthly charges")
    axes[1].set_title("Monthly charges by churn status")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tenure_and_charges.png", dpi=150)
    plt.close(fig)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
    else:
        scores = predictions

    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "pr_auc": float(average_precision_score(y_test, scores)),
    }
    return metrics, model, predictions


def save_confusion_matrices(y_test, predictions_by_model):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (name, predictions) in zip(axes.ravel(), predictions_by_model.items()):
        matrix = confusion_matrix(y_test, predictions)
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix, display_labels=["No churn", "Churn"]
        )
        display.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = clean_data(load_data())
    churn_rate = df["Churn"].mean()
    print("Shape:", df.shape)
    print(f"Churn rate: {churn_rate:.1%}")
    print("\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print("\nNumeric summary:")
    print(df.describe().round(2))

    save_exploration_plots(df, churn_rate)

    X = df.drop(columns="Churn")
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    classifiers = {
        "Baseline (prior)": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
        ),
    }

    results = []
    fitted_models = {}
    predictions_by_model = {}
    for name, classifier in classifiers.items():
        pipeline = Pipeline(
            [("preprocessor", build_preprocessor(X_train)), ("classifier", classifier)]
        )
        metrics, fitted, predictions = evaluate_model(
            name, pipeline, X_train, X_test, y_train, y_test
        )
        results.append(metrics)
        fitted_models[name] = fitted
        predictions_by_model[name] = predictions

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    print("\nModel comparison (sorted by F1 on the churn class):")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    save_confusion_matrices(y_test, predictions_by_model)

    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name}")
    print("\nClassification report for the best model:")
    print(
        classification_report(
            y_test,
            predictions_by_model[best_name],
            target_names=["No churn", "Churn"],
            digits=3,
        )
    )

    model_path = MODEL_DIR / "churn_model.joblib"
    joblib.dump(best_model, model_path)
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Saved best model to {model_path}")
    print(f"Saved plots and metrics to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
