"""SMS spam classification with Naive Bayes and model comparison.

Run from the repository root:
    python projects/spam-classification/train.py

The SMS Spam Collection dataset is downloaded once and cached in data/raw/. If
the download is unavailable, a small embedded sample is used instead.
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
SMS_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
    "master/data/sms.tsv"
)
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data" / "raw"

FALLBACK_MESSAGES = [
    ("ham", "Go until jurong point, crazy.. Available only in bugis n great world"),
    ("ham", "Ok lar... Joking wif u oni..."),
    ("ham", "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight"),
    ("ham", "Did you catch the game last night? What a finish!"),
    ("ham", "Can you send me the notes from today's lecture?"),
    ("ham", "Are we still meeting for lunch tomorrow at noon?"),
    ("ham", "The project deadline has moved to next Friday."),
    ("ham", "Thanks for your help earlier, I really appreciate it."),
    ("ham", "I'll be a bit late, traffic is terrible."),
    ("ham", "Happy birthday! Hope you have a wonderful day."),
    ("ham", "Please review the attached report before the meeting."),
    ("ham", "Let me know when you get home so I know you're safe."),
    ("spam", "WINNER!! As a valued network customer you have been selected to receive a prize"),
    ("spam", "Free entry in 2 a weekly competition to win FA Cup final tickets"),
    ("spam", "URGENT! You have won a 1 week FREE membership in our prize draw"),
    ("spam", "Congratulations! You have won a 1000 cash prize. Call now to claim"),
    ("spam", "Claim your free iPhone now, limited stock available"),
    ("spam", "Low interest loans available. Apply today with no credit check"),
    ("spam", "Text WIN to 80086 to receive your free ringtone"),
    ("spam", "You have been selected for a special discount. Buy now and save 70%"),
    ("spam", "Cash bonus waiting for you. Reply YES to claim your reward"),
    ("spam", "Final notice: your account will be suspended unless you verify now"),
    ("spam", "Earn money from home, no experience needed. Register today"),
    ("spam", "Exclusive offer just for you, click the link to get started"),
]


def load_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / "sms_spam.tsv"
    if cache_path.exists():
        print(f"Loading cached dataset from {cache_path}")
        df = pd.read_csv(cache_path, sep="\t", header=None, names=["label", "message"])
        return df

    try:
        df = pd.read_csv(SMS_URL, sep="\t", header=None, names=["label", "message"])
        df.to_csv(cache_path, sep="\t", index=False, header=False)
        print(f"Downloaded dataset and cached it at {cache_path}")
        return df
    except Exception as exc:  # network failures should not block the lesson
        print(f"Could not download the SMS Spam Collection ({exc}).")
        print("Falling back to a small embedded sample of SMS messages.")
        return pd.DataFrame(FALLBACK_MESSAGES, columns=["label", "message"])


def clean_data(df):
    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])]
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"] != ""].drop_duplicates(subset="message")
    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    df["length"] = df["message"].str.len()
    return df.reset_index(drop=True)


def save_exploration_plots(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = df["label"].value_counts()
    bars = axes[0].bar(counts.index, counts.values, color=["#4c72b0", "#c44e52"])
    axes[0].bar_label(bars)
    axes[0].set_ylabel("Messages")
    axes[0].set_title("Class balance")

    for label, name, color in [(0, "ham", "#4c72b0"), (1, "spam", "#c44e52")]:
        axes[1].hist(
            df[df["target"] == label]["length"],
            bins=40,
            alpha=0.6,
            label=name,
            color=color,
        )
    axes[1].set_xlabel("Message length (characters)")
    axes[1].set_ylabel("Messages")
    axes[1].set_title("Message length by class")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_balance_and_lengths.png", dpi=150)
    plt.close(fig)


def build_pipeline(classifier):
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "pr_auc": float(average_precision_score(y_test, scores)),
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
    }
    return metrics, model, predictions


def save_confusion_matrices(y_test, predictions_by_model):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, predictions) in zip(axes, predictions_by_model.items()):
        matrix = confusion_matrix(y_test, predictions)
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix, display_labels=["Ham", "Spam"]
        )
        display.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def save_top_tokens(model):
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    if not isinstance(classifier, MultinomialNB):
        return

    feature_names = np.asarray(vectorizer.get_feature_names_out())
    log_difference = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
    order = np.argsort(log_difference)

    top_ham = [(feature_names[i], log_difference[i]) for i in order[:15]]
    top_spam = [(feature_names[i], log_difference[i]) for i in order[-15:][::-1]]

    print("\nTop tokens for ham (negative -> strong ham):")
    for token, score in top_ham:
        print(f"  {token:<25} {score:+.3f}")
    print("\nTop tokens for spam (positive -> strong spam):")
    for token, score in top_spam:
        print(f"  {token:<25} {score:+.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, tokens, title, color in [
        (axes[0], top_spam, "Strongest spam tokens", "#c44e52"),
        (axes[1], top_ham, "Strongest ham tokens", "#4c72b0"),
    ]:
        labels = [token for token, _ in tokens][::-1]
        values = [score for _, score in tokens][::-1]
        ax.barh(labels, values, color=color)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("log P(token | spam) - log P(token | ham)")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_tokens.png", dpi=150)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = clean_data(load_data())
    print("Shape:", df.shape)
    print("\nClass distribution:")
    print(df["label"].value_counts().to_string())
    print("\nAverage message length by class:")
    print(df.groupby("label")["length"].mean().round(1).to_string())

    save_exploration_plots(df)

    X = df["message"]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    priors = np.bincount(y_train, minlength=2) / len(y_train)
    print(f"\nTraining priors: P(ham)={priors[0]:.3f}, P(spam)={priors[1]:.3f}")

    classifiers = {
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
    }

    results = []
    fitted_models = {}
    predictions_by_model = {}
    for name, classifier in classifiers.items():
        model = build_pipeline(classifier)
        metrics, fitted, predictions = evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )
        results.append(metrics)
        fitted_models[name] = fitted
        predictions_by_model[name] = predictions

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    print("\nModel comparison (sorted by test F1):")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    save_confusion_matrices(y_test, predictions_by_model)
    save_top_tokens(fitted_models["Naive Bayes"])

    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name}")
    print("\nClassification report for the best model:")
    print(
        classification_report(
            y_test,
            predictions_by_model[best_name],
            target_names=["Ham", "Spam"],
            digits=3,
        )
    )

    model_path = MODEL_DIR / "spam_model.joblib"
    joblib.dump(best_model, model_path)
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Saved best model to {model_path}")
    print(f"Saved plots and metrics to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
