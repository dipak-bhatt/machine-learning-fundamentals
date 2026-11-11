# Day 20 — Theory: Preprocessing, Hyperparameter Tuning, and Responsible ML

**Module:** Module 5: Evaluation and Optimization

Real projects rarely feed raw columns straight into a model. Numeric features need
imputation and scaling, categorical features need encoding, and the whole sequence
must be applied identically at training and prediction time. Doing this by hand is
error-prone and is one of the most common sources of leakage.

A **pipeline** packages preprocessing and the model into one object that can be
cross-validated and tuned as a unit. **Hyperparameter search** then explores
configurations without touching the test data. This day covers both, plus the
responsible-use questions that decide whether a model should be deployed at all.

## In Plain English

**Think of it like this:** a pipeline glues preprocessing and the model into one
machine, so the same steps always run in the same order; tuning then searches
settings without peeking at the test set.

- **Pipeline** prevents leakage and keeps training and prediction consistent.
- **ColumnTransformer** sends numeric and categorical columns down different
  preprocessing paths.
- **Grid search / random search** try many hyperparameter combinations using
  cross-validation.
- Fit every learned preprocessing step **inside each fold**, never on all data.
- Report final performance once, and also consider fairness, privacy, and
  monitoring.

## 1. Why Manual Preprocessing Breaks

A typical manual workflow looks like this:

```python
X = df.drop(columns="target")
X = X.fillna(X.median())
X = pd.get_dummies(X)
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y)
```

It works on a single split, but it is fragile:

- The imputer, encoder, and scaler saw the test rows before the split.
- At prediction time, new categories and different missing patterns may break the
  code.
- Cross-validation cannot reproduce the same steps for each fold.
- The model and its preprocessing can drift apart across scripts.

The fix is to make preprocessing part of the model.

## 2. The Pipeline Abstraction

A scikit-learn `Pipeline` is a sequence of named steps ending in an estimator:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression()),
])
```

The pipeline exposes the estimator interface:

- `fit(X, y)` calls `fit_transform` on each transformer in order, then `fit` on the
  final estimator.
- `predict(X)` calls `transform` on each transformer, then `predict` on the final
  estimator.
- `score(X, y)` uses the final estimator's scoring.
- `predict_proba`, `decision_function`, and other methods are forwarded.

Two consequences matter:

1. **Every learned transformation is fit on the training data only.** During CV, each
   fold refits the whole chain.
2. **The same chain is used at inference.** There is no separate preprocessing script
   to keep in sync.

Steps are named so their parameters can be addressed with the `step__parameter`
syntax, for example `classifier__C`.

## 3. ColumnTransformer

Different columns need different treatment. `ColumnTransformer` applies separate
pipelines to selected columns and concatenates the results.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = ["tenure", "monthly_charge", "age"]
categorical_features = ["plan", "region"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression()),
])
```

`handle_unknown="ignore"` is important: at prediction time a category that was not in
training becomes all zeros instead of raising an error. Without it, a new region or
plan can crash a deployed model.

## 4. Preprocessing Steps in Detail

### Missing-value imputation

Models generally cannot handle `NaN`. Common strategies:

- **Mean or median** for numeric features. Median is robust to outliers.
- **Most frequent** for categorical features.
- **Constant** such as `"missing"`, which preserves missingness as information.
- **Model-based** imputation (KNN, iterative) when relationships are strong.

Missingness itself can be informative. In some domains, "not recorded" correlates
with the outcome. Consider adding a missing-indicator column rather than imputing it
away.

### Categorical encoding

- **One-hot encoding** creates one binary column per category. Good for low-
  cardinality features; expensive for high-cardinality ones.
- **Ordinal encoding** maps categories to integers, appropriate only when order is
  meaningful.
- **Target encoding** replaces a category with the mean target for that category.
  Powerful for high cardinality but risky: it can leak the target if computed outside
  the CV fold.
- **Hashing** maps many categories to a fixed number of columns, trading
  interpretability for scalability.

### Scaling

- **StandardScaler**: mean 0, variance 1. Default for linear models, SVM, KNN, PCA.
- **MinMaxScaler**: maps to a fixed range. Useful when a bounded range matters.
- **RobustScaler**: uses the median and interquartile range, resistant to outliers.

Tree-based models do not need scaling, but including it is usually harmless and keeps
the pipeline uniform.

### Feature selection and engineering

Selection and engineering are preprocessing too, and they must live inside the
pipeline when they learn from data. Examples:

- `SelectKBest`, `RFE`, `SelectFromModel`.
- Polynomial features, interaction terms, ratios, date parts.
- Binning continuous variables.

Feature engineering based on domain knowledge and applied row-by-row is safe to do
before the split. Anything that estimates a statistic from the data is not.

## 5. Parameters versus Hyperparameters

A **parameter** is learned during `fit`: coefficients, tree splits, support vectors.
A **hyperparameter** is set before `fit`: regularization strength `C`, number of
neighbors `k`, tree depth, learning rate.

Hyperparameters control model capacity and must be chosen by evaluation. Tuning is the
process of searching for a good configuration using validation data, never the test
set.

Some preprocessing choices are also hyperparameters: imputation strategy, number of
features to select, or whether to scale. Because the pipeline exposes every step's
parameters, they can be tuned together with the model.

## 6. Grid Search

**Grid search** evaluates every combination from a predefined grid.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__solver": ["lbfgs", "liblinear"],
}

search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
search.fit(X_train, y_train)

print(search.best_params_)
print(search.best_score_)
```

Properties:

- **Exhaustive** and easy to reason about.
- Cost grows multiplicatively with the number of values.
- Poor use of budget when some hyperparameters matter far more than others.
- Best when the grid is small and informed by domain knowledge.

`best_score_` is the mean cross-validation score of the best configuration. It is
still a validation estimate, not a test score.

## 7. Randomized Search

**Randomized search** samples a fixed number of configurations from distributions.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

param_distributions = {
    "classifier__C": loguniform(1e-3, 1e3),
}

search = RandomizedSearchCV(
    model,
    param_distributions,
    n_iter=30,
    cv=5,
    scoring="accuracy",
    random_state=42,
)
search.fit(X_train, y_train)
```

Properties:

- Covers a wider range with a fixed budget.
- Often finds a good configuration faster than grid search when only a few
  hyperparameters matter.
- Results are stochastic unless `random_state` is fixed.
- Ideal when the search space is large or continuous.

Research on random search shows that for many problems it matches or beats grid
search at a fraction of the cost, because most hyperparameters have little effect and
only a few matter.

## 8. Choosing the Search Strategy

Consider:

- **Budget**: how many fits can you afford? Grid cost is the product of grid sizes;
  random cost is `n_iter` times CV folds.
- **Dimensionality**: grids explode with many hyperparameters. Prefer random or
  Bayesian optimization.
- **Prior knowledge**: if you know the useful range, a small grid is efficient.
- **Reproducibility**: fix `random_state` and record the search space.

Whatever the strategy, the score is only as meaningful as the CV scheme. Use
stratified folds for classification, groups when rows are not independent, and
time-ordered splits for temporal data.

## 9. Selection Bias and Nested CV

If you tune hyperparameters on the same CV folds you use to report performance, the
reported score is optimistic. The search has seen the validation folds and chosen the
configuration that happened to do well on them.

**Nested cross-validation** separates the two loops:

- **Inner loop**: tunes hyperparameters on the training part of each outer fold.
- **Outer loop**: evaluates the chosen model on the held-out part.

```text
outer fold 1: [ test ][ -------- train -------- ]
                        inner CV selects hyperparameters
outer fold 2: [ -------- train -------- ][ test ]
...
```

The mean outer score is a less biased estimate of the modeling *procedure*, not of one
fixed configuration. It is more expensive but appropriate when the reported number
must be trustworthy.

In practice, many teams use a single holdout test set that is never touched during
tuning, which is simpler and nearly as honest. The key rule is the same: tuning and
final evaluation must use different data.

## 10. Resampling Inside the Pipeline

Class imbalance is often handled by oversampling, undersampling, or SMOTE. These
operations must happen **inside** the CV loop, on the training part only.

If you resample before splitting:

- Synthetic or duplicated examples can appear in both train and test.
- The test set no longer reflects the real distribution.
- Scores become optimistic.

Scikit-learn's `Pipeline` can include a sampler in recent versions, or you can use
`imblearn.pipeline.Pipeline`, which is designed for resampling steps. The principle is
the same as for scaling: fit the sampler on training data only.

Also remember that resampling changes the base rate. Predicted probabilities are no
longer calibrated to the original population, so thresholds may need adjustment.

## 11. Responsible ML

A model that scores well can still cause harm. Responsible practice asks four
questions.

### Fairness

Does the model perform similarly across groups defined by sensitive attributes such as
age, sex, or region? Aggregate accuracy can hide large disparities. Measure metrics
per group, and decide explicitly which fairness notion matters:

- **Demographic parity**: similar positive rates across groups.
- **Equal opportunity**: similar true positive rates across groups.
- **Equalized odds**: similar TPR and FPR across groups.

These criteria can conflict. There is no purely technical answer; the choice is a
policy decision.

### Privacy

Models can memorize or leak training data. Risks include:

- Re-identification from supposedly anonymous records.
- Membership inference: determining whether a person was in the training set.
- Reconstruction of sensitive attributes.

Mitigations include minimizing data, aggregating, adding noise (differential
privacy), and avoiding logging sensitive fields. Never put raw personal data in logs
or model artifacts.

### Explainability

Stakeholders need to understand and challenge decisions. Options range from inherently
interpretable models (linear models, small trees) to post-hoc explanations (feature
importance, SHAP, partial dependence).

Explanations are approximations. They can be unstable and should not be treated as
causal proof. For high-stakes decisions, favor interpretable models or require human
review.

### Accountability and appropriate use

- Document the intended use, training data, and known limitations.
- Monitor performance and data drift after deployment.
- Provide a path for appeal or correction.
- Refuse uses outside the validated scope.
- Consider the cost of errors for the people affected, not just the organization.

A model is a tool embedded in a decision process. The process, not just the model,
determines whether the outcome is responsible.

## 12. Common Mistakes

- **Preprocessing before the split or outside CV.** Leakage.
- **Tuning on the test set.** Inflated and unusable estimates.
- **Reporting `best_score_` as test performance.** It is a validation score.
- **Forgetting `handle_unknown="ignore"`.** New categories crash production.
- **Imputing without recording why values were missing.** Missingness can be signal.
- **Resampling before splitting.** Synthetic leakage.
- **Ignoring group structure.** The same entity in train and test inflates scores.
- **Choosing a huge grid.** Cost grows fast with little gain.
- **Comparing models tuned with different CV schemes.** The comparison is not fair.
- **Skipping subgroup evaluation.** Aggregate metrics hide harm.
- **Deploying without monitoring.** Data drift silently degrades models.

## 13. Summary

- A pipeline packages preprocessing and the model so they are fit and applied together.
- `ColumnTransformer` handles numeric and categorical columns differently.
- Imputation, encoding, scaling, selection, and engineering are all preprocessing.
- Parameters are learned; hyperparameters are chosen by evaluation.
- Grid search is exhaustive; randomized search spends a fixed budget more flexibly.
- Tuning and final evaluation must use different data; nested CV formalizes this.
- Resampling belongs inside the CV loop.
- Responsible ML requires attention to fairness, privacy, explainability, and scope.
- Subgroup metrics and documentation are part of evaluation, not extras.

## Outcome

Learners can build reproducible pipelines with `Pipeline` and `ColumnTransformer`,
apply imputation, encoding, and scaling correctly, distinguish parameters from
hyperparameters, run grid and randomized search inside cross-validation, explain and
avoid selection bias, place resampling inside the pipeline, and evaluate a model for
fairness, privacy, and responsible use.
