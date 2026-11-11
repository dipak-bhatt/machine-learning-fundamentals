# Day 21 — Theory: End-to-End ML Pipeline

**Module:** Module 6: Modern ML and Project

A model that exists only in a notebook is not a solution. Real work is a lifecycle:
define the problem, collect and split data, preprocess, train, evaluate, save,
serve, and monitor. Most failures happen between the steps, not inside the algorithm.
This day treats the workflow as a single reproducible pipeline rather than a
collection of disconnected cells.

## In Plain English

**Think of it like this:** an ML project is a loop, not a one-time notebook —
define, build, ship, watch, and improve.

- Start from a **business question** and a measurable success metric.
- Data collection and cleaning usually take the most time.
- Version your **data, code, and models** so results are reproducible.
- **Deploy**, monitor for drift, and retrain when the world changes.
- Treat the model as a **product** with users and maintenance, not a one-off
  experiment.

## 1. The Lifecycle

```text
problem definition
        |
data collection and cleaning
        |
train / validation / test split
        |
preprocessing (inside a pipeline)
        |
baseline model
        |
training and model selection
        |
final evaluation on the test set
        |
save, version, and deploy
        |
monitor, retrain, repeat
```

Each stage answers a question:

- **Problem definition**: what decision does the model support, and how is success
  measured?
- **Data**: what is available, how was it collected, and what are its known biases?
- **Split**: how do we estimate performance on data we have not seen?
- **Preprocessing**: how do raw columns become model inputs, consistently?
- **Baseline**: what is the simplest reasonable solution?
- **Training**: which model and hyperparameters?
- **Evaluation**: how well does it generalize?
- **Deployment**: how does it reach users, and how is it kept healthy?

Skipping a stage does not remove the work; it moves the failure later, where it is
more expensive.

## 2. Problem Definition

Before any modeling, write down:

- **Prediction target**: what exactly is being predicted?
- **Unit of prediction**: one row, one user, one transaction, one time step?
- **When the prediction is made**: what information is available at that moment?
- **Metric**: what number represents success?
- **Baseline**: what does "no model" achieve?
- **Constraints**: latency, cost, interpretability, privacy, fairness.
- **Failure cost**: what happens when the model is wrong, and in which direction?

A useful check is to describe the deployed system in one sentence: "Given a house's
area, age, and location, predict its sale price so that a listing tool can flag
unrealistic prices." If you cannot say this, the problem is not yet defined.

### Point-in-time correctness

The features available at prediction time may differ from those available at training
time. Using future information, such as the final sale price or a later status field,
creates **temporal leakage**. For each feature, ask: "Would I actually know this when
the prediction is made?"

## 3. Data Collection and Splitting

Split before inspecting performance. The standard arrangement:

- **Training set**: fit parameters and preprocessing.
- **Validation set**: compare models and tune hyperparameters.
- **Test set**: touched once, at the end, for the final estimate.

With small data, cross-validation replaces a fixed validation set, but the test set
still stays untouched.

The split must respect the data structure:

- **Classification**: stratify to preserve class proportions.
- **Groups**: keep all rows from the same entity in one split.
- **Time**: train on the past, test on the future.
- **Duplicates**: remove or group them so the same record does not appear in both
  sets.

Report the split sizes and the class or target distribution. An accidental imbalance
between train and test can explain a surprising score.

## 4. Preprocessing

Preprocessing is part of the model. It must be fit on the training data and reused
unchanged for validation, test, and production. A `Pipeline` with a
`ColumnTransformer` is the standard way to guarantee this:

```python
model = Pipeline([
    ("preprocess", ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ])),
    ("regressor", GradientBoostingRegressor(random_state=42)),
])
```

Because the whole object is fitted and saved as one unit, the training and serving
transformations cannot drift apart. That property is called **training-serving
consistency**.

## 5. Baseline Models

Always start with a baseline. A baseline is the simplest model that could plausibly
work, and it turns a raw score into a meaningful comparison.

For regression:

- Predict the **mean** or **median** of the training target.
- Predict the value of a single strong feature.

For classification:

- Predict the **majority class**.
- Predict using one interpretable rule.

For a baseline, the expected score is often:

- Regression: `R² ≈ 0`.
- Classification: accuracy equal to the majority-class rate.

If a complex model barely beats the baseline, the problem may be close to noise, the
features may be weak, or the evaluation may be broken. A strong baseline is a useful
alarm.

## 6. Training and Model Selection

Train several candidate models and compare them on validation data or cross-
validation. Keep the comparison fair:

- Same splits for every candidate.
- Same preprocessing protocol, fit inside the fold.
- Same metric and averaging.
- Same random seeds.

Select for the metric that matches the decision. A model with the best average score
may not be the best choice if it is unstable, slow, or hard to explain.

For small datasets, prefer simpler models. They are easier to validate, cheaper to
maintain, and less likely to overfit. Complexity should earn its place with a real
improvement.

## 7. Final Evaluation

Evaluate the selected model **once** on the test set. Report:

- The primary metric.
- Secondary metrics that reveal trade-offs.
- The confidence interval or fold-to-fold variation.
- The baseline score for context.
- The confusion matrix or residual summary.

If you evaluate many models on the test set and pick the best, the test set has become
a validation set and the estimate is optimistic. That is why the split exists.

## 8. Reproducibility

A result is reproducible when another person can obtain the same number from the same
data and code. Practical requirements:

- **Fixed seeds** for every stochastic step: splits, model initialization, sampling.
- **Pinned versions** of libraries.
- **Versioned data**, or at least a recorded snapshot and hash.
- **Code in version control**, with the exact commit.
- **Recorded configuration**, including hyperparameters and preprocessing choices.
- **A single command** or notebook that runs the whole pipeline.

Seeds do not make a model deterministic across library versions or hardware, but they
remove avoidable variation.

```python
RANDOM_STATE = 42
```

Use one constant everywhere, and pass it explicitly. Do not rely on global state.

## 9. Saving and Loading Models

A fitted pipeline can be serialized to disk:

```python
import joblib

joblib.dump(model, "house_price_model.joblib")
loaded = joblib.load("house_price_model.joblib")

predictions = loaded.predict(new_records)
```

`joblib` is efficient for NumPy-heavy objects and is the common choice for
scikit-learn. `pickle` works too but is less efficient for large arrays.

Important caveats:

- **Version compatibility.** A model saved with one library version may not load with
  another. Pin versions in production.
- **Security.** Loading a pickle or joblib file executes code. Never load models from
  untrusted sources.
- **Save the whole pipeline**, not just the estimator, so preprocessing travels with
  it.
- **Record metadata** alongside the file: training date, data version, metric,
  package versions, and intended use.

## 10. Model Versioning and Experiment Tracking

As models change, you need to know which one is deployed and why.

**Model versioning** assigns an identifier to each trained artifact. A version should
be traceable to:

- The training data snapshot.
- The code commit.
- The hyperparameters.
- The evaluation results.
- The date and owner.

**Experiment tracking** records each run: parameters, metrics, artifacts, and notes.
Tools range from a spreadsheet to MLflow, Weights & Biases, or a database. The
principle matters more than the tool: every result should be reproducible and
comparable.

A simple naming scheme is better than none:

```text
house-price-gbr-2026-11-10-v3
```

## 11. Monitoring and Retraining

A model's performance can decay even if nothing in the code changes. Causes include:

- **Data drift**: input distributions shift.
- **Concept drift**: the relationship between inputs and target changes.
- **Pipeline drift**: an upstream data source changes format or meaning.
- **Feedback loops**: the model's own decisions alter future data.

Monitor:

- Input distributions and missing rates.
- Prediction distributions.
- Live performance once labels arrive.
- Error rates by segment.
- Latency and availability.

Retrain when drift is detected or on a schedule, and re-run the full evaluation. A
retrained model is a new model and needs the same scrutiny.

## 12. Common Mistakes

- **Starting with a complex model.** No baseline, no context.
- **Tuning on the test set.** Optimistic and unrecoverable.
- **Preprocessing outside the pipeline.** Training-serving drift and leakage.
- **Not fixing seeds.** Results cannot be reproduced.
- **Saving only the estimator.** Preprocessing is lost or reimplemented differently.
- **No version metadata.** Nobody knows which model is live.
- **Ignoring point-in-time correctness.** Future information leaks into features.
- **Judging by one metric.** Trade-offs and subgroup performance are hidden.
- **Deploying without monitoring.** Silent degradation.
- **Trusting a model outside its validated scope.** The most common real-world
  failure.

## 13. Summary

- The ML lifecycle spans problem definition through monitoring.
- Define the target, unit, timing, metric, baseline, and constraints first.
- Split before evaluating, and respect groups and time.
- Preprocessing is part of the model and must be fit inside the pipeline.
- Always establish a baseline.
- Compare candidates on the same splits, metric, and seeds.
- Evaluate the test set once.
- Reproducibility requires seeds, pinned versions, versioned data, and recorded
  configuration.
- Save the entire pipeline and attach metadata.
- Version and track experiments so results are traceable.
- Monitor for drift and retrain deliberately.

## Outcome

Learners can describe the full ML lifecycle, define a problem and baseline, build a
leak-free reproducible pipeline, compare models fairly, evaluate once on held-out
data, save and load complete pipelines, record versions and experiments, and plan for
monitoring and retraining.
