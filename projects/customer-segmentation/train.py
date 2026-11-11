"""Customer segmentation with K-means, the elbow method, and silhouette scores.

Run from the repository root:
    python projects/customer-segmentation/train.py

The Mall Customers dataset is downloaded once and cached in data/raw/. If the
download is unavailable, a deterministic synthetic customer dataset is used.
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
MALL_URL = (
    "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/"
    "Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/"
    "Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv"
)
FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data" / "raw"


def make_synthetic_customers(n_samples=200, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    centers = [
        (25, 25, 80),
        (32, 25, 20),
        (45, 55, 50),
        (50, 85, 80),
        (52, 85, 15),
    ]
    weights = [0.20, 0.20, 0.25, 0.18, 0.17]
    sizes = rng.multinomial(n_samples, weights)

    ages, incomes, scores = [], [], []
    for (age, income, score), size in zip(centers, sizes):
        ages.append(rng.normal(age, 6, size))
        incomes.append(rng.normal(income, 10, size))
        scores.append(rng.normal(score, 10, size))

    df = pd.DataFrame(
        {
            "CustomerID": np.arange(1, n_samples + 1),
            "Genre": rng.choice(["Male", "Female"], n_samples),
            "Age": np.clip(np.round(np.concatenate(ages)), 18, 70).astype(int),
            "Annual Income (k$)": np.clip(
                np.round(np.concatenate(incomes)), 15, 137
            ).astype(int),
            "Spending Score (1-100)": np.clip(
                np.round(np.concatenate(scores)), 1, 99
            ).astype(int),
        }
    )
    return df


def load_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / "mall_customers.csv"
    if cache_path.exists():
        print(f"Loading cached dataset from {cache_path}")
        return pd.read_csv(cache_path)

    try:
        df = pd.read_csv(MALL_URL)
        df.to_csv(cache_path, index=False)
        print(f"Downloaded dataset and cached it at {cache_path}")
        return df
    except Exception as exc:  # network failures should not block the lesson
        print(f"Could not download the Mall Customers dataset ({exc}).")
        print("Falling back to a deterministic synthetic customer dataset.")
        return make_synthetic_customers()


def clean_data(df):
    missing = [feature for feature in FEATURES if feature not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df.copy()
    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
    df = df.dropna(subset=FEATURES).drop_duplicates()
    return df.reset_index(drop=True)


def save_exploration_plots(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, feature in zip(axes, FEATURES):
        ax.hist(df[feature], bins=20, color="#4c72b0", edgecolor="white")
        ax.set_xlabel(feature)
        ax.set_ylabel("Customers")
        ax.set_title(f"Distribution of {feature}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        s=25,
        alpha=0.7,
        color="#4c72b0",
    )
    ax.set_xlabel("Annual income (k$)")
    ax.set_ylabel("Spending score (1-100)")
    ax.set_title("Customers before clustering")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "customers_raw.png", dpi=150)
    plt.close(fig)


def choose_clusters(X_scaled, k_min=2, k_max=10):
    results = []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = model.fit_predict(X_scaled)
        results.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(X_scaled, labels)),
            }
        )
    return pd.DataFrame(results)


def save_cluster_selection_plot(selection):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(selection["k"], selection["inertia"], marker="o", color="#4c72b0")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow method")
    axes[0].grid(alpha=0.3)

    axes[1].plot(selection["k"], selection["silhouette"], marker="o", color="#55a868")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette score by k")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cluster_selection.png", dpi=150)
    plt.close(fig)


def save_cluster_plot(df, labels, centers, best_k):
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=labels,
        cmap="tab10",
        s=30,
        alpha=0.7,
    )
    ax.scatter(
        centers[:, 1],
        centers[:, 2],
        marker="X",
        s=250,
        c="black",
        edgecolor="white",
        linewidth=1.5,
        label="Centroids",
    )
    ax.set_xlabel("Annual income (k$)")
    ax.set_ylabel("Spending score (1-100)")
    ax.set_title(f"Customer segments (k={best_k})")
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "customer_segments.png", dpi=150)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = clean_data(load_data())
    print("Shape:", df.shape)
    print("\nSummary statistics:")
    print(df[FEATURES].describe().round(2))
    save_exploration_plots(df)

    X = df[FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selection = choose_clusters(X_scaled)
    print("\nCluster selection:")
    print(selection.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    save_cluster_selection_plot(selection)

    best_k = int(selection.loc[selection["silhouette"].idxmax(), "k"])
    print(f"\nBest k by silhouette score: {best_k}")

    model = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
    labels = model.fit_predict(X_scaled)
    centers = scaler.inverse_transform(model.cluster_centers_)

    print(f"Inertia: {model.inertia_:.2f}")
    print(f"Silhouette: {silhouette_score(X_scaled, labels):.3f}")

    save_cluster_plot(df, labels, centers, best_k)

    profile = df.copy()
    profile["Cluster"] = labels
    summary = (
        profile.groupby("Cluster")[FEATURES]
        .mean()
        .round(2)
        .assign(Customers=profile["Cluster"].value_counts().sort_index())
    )
    print("\nCluster profiles (mean feature values):")
    print(summary.to_string())

    profile.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "cluster_profiles.csv")

    model_path = MODEL_DIR / "segmentation_model.joblib"
    joblib.dump(
        {"model": model, "scaler": scaler, "features": FEATURES, "k": best_k},
        model_path,
    )
    with open(OUTPUT_DIR / "cluster_selection.json", "w", encoding="utf-8") as handle:
        json.dump(selection.to_dict(orient="records"), handle, indent=2)

    print(f"\nSaved model to {model_path}")
    print(f"Saved plots and segment tables to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
