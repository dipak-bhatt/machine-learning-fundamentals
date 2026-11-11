# Day 11 — Practical: Ensemble Learning

**Module:** Module 2: Supervised Learning

In this practical we combine many models and measure the result honestly. We will:

- Compare a dummy baseline, logistic regression, and a single tree.
- Train a random forest and inspect its out-of-bag score.
- Train gradient boosting and histogram gradient boosting.
- Compare all models on one identical train/test split.
- Use cross-validation to see fold-to-fold variability.
- Study `n_estimators`, `learning_rate`, and their interaction.
- Compare impurity-based and permutation feature importance.
- Combine models with voting and stacking.

Everything uses the breast cancer data bundled with scikit-learn, so the notebook
can live in this Day 11 folder with no path configuration.

## 1. Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score

RANDOM_STATE = 42
```

The seed fixes the split and the randomness inside the ensembles. Forests and
boosting are stochastic, so repeatability matters when comparing runs.

## 2. Load and Split the Data

```python
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

print("train:", X_train.shape, "test:", X_test.shape)
print("train classes:", np.bincount(y_train))
print("test classes:", np.bincount(y_test))
```

Expected output:

```text
train: (426, 30) test: (143, 30)
train classes: [159 267]
test classes: [ 53  90]
```

The classes are moderately imbalanced, with benign cases more common. Stratifying
keeps that ratio in both parts.

## 3. Baselines

Always start with the simplest possible predictor and one strong simple model. The
dummy model always predicts the majority class. Logistic regression gets a scaler
so it converges cleanly.

```python
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)

logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
logistic.fit(X_train, y_train)

print("dummy:", round(dummy.score(X_test, y_test), 4))
print("logistic:", round(logistic.score(X_test, y_test), 4))
```

Expected output:

```text
dummy: 0.6294
logistic: 0.986
```

The dummy model gets 63% by always predicting benign. Any real model must clear
that bar. Logistic regression clears it easily, which already tells us the classes
are close to linearly separable in this feature space. Keep that in mind when the
ensembles arrive; beating 98.6% is hard.

## 4. A Single Decision Tree

The tree is the base learner we will improve on. Leave it unconstrained to show its
instability.

```python
tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
tree.fit(X_train, y_train)

print("train:", round(tree.score(X_train, y_train), 4))
print("test:", round(tree.score(X_test, y_test), 4))
print("depth:", tree.get_depth(), "leaves:", tree.get_n_leaves())
```

Expected output:

```text
train: 1.0
test: 0.9231
depth: 19
leaves: 40
```

The tree memorizes the training set and loses about seven points on test data. It
is a high-variance learner, exactly the kind that bagging helps.

## 5. Random Forest

A forest trains many deep trees on bootstrap samples and random feature subsets.

```python
forest = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
forest.fit(X_train, y_train)

print("train:", round(forest.score(X_train, y_train), 4))
print("test:", round(forest.score(X_test, y_test), 4))
```

Expected output:

```text
train: 1.0
test: 0.958
```

The forest still fits the training data perfectly, but test accuracy rises from
92.3% to 95.8%. The individual trees are just as overfit as before; averaging their
votes cancels part of their variance.

### Out-of-bag score

Set `oob_score=True` to estimate generalization from the rows each tree did not
see. Compare different forest sizes.

```python
for n_trees in [10, 50, 100, 300]:
    model = RandomForestClassifier(
        n_estimators=n_trees,
        oob_score=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(
        f"trees={n_trees}: train={model.score(X_train, y_train):.4f} "
        f"test={model.score(X_test, y_test):.4f} "
        f"oob={model.oob_score_:.4f}"
    )
```

Expected output:

```text
trees=10: train=1.0000 test=0.9510 oob=0.9484
trees=50: train=1.0000 test=0.9510 oob=0.9601
trees=100: train=1.0000 test=0.9580 oob=0.9624
trees=300: train=1.0000 test=0.9580 oob=0.9624
```

Two lessons:

- The OOB score is close to the test score, which is why it is useful when a
  separate validation set is expensive.
- Returns flatten after a few hundred trees. Adding more trees rarely hurts, but it
  costs time and memory.

With only ten trees, scikit-learn warns that some rows have too few OOB
predictions. Treat the OOB estimate with caution for small forests.

## 6. Gradient Boosting

Gradient boosting builds shallow trees in sequence, each one correcting the errors
of the ensemble so far.

```python
boosting = GradientBoostingClassifier(random_state=RANDOM_STATE)
boosting.fit(X_train, y_train)

print("train:", round(boosting.score(X_train, y_train), 4))
print("test:", round(boosting.score(X_test, y_test), 4))
print("trees:", boosting.n_estimators, "learning rate:", boosting.learning_rate)
```

Expected output:

```text
train: 1.0
test: 0.958
```

Histogram gradient boosting is a faster variant that bins features. It often
matches or beats the classic implementation.

```python
hist_boosting = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
hist_boosting.fit(X_train, y_train)

print("train:", round(hist_boosting.score(X_train, y_train), 4))
print("test:", round(hist_boosting.score(X_test, y_test), 4))
```

Expected output:

```text
train: 1.0
test: 0.965
```

On this dataset the histogram version is slightly ahead. That is not a guarantee;
it depends on the data size and feature distributions.

## 7. Same Split, Side by Side

Comparing models on different splits is meaningless. Put every result in one table
built from the same `X_train` and `X_test`.

```python
models = {
    "dummy": dummy,
    "logistic": logistic,
    "tree": tree,
    "random forest": forest,
    "gradient boosting": boosting,
    "hist gradient boosting": hist_boosting,
}

rows = []
for name, model in models.items():
    rows.append(
        {
            "model": name,
            "train": model.score(X_train, y_train),
            "test": model.score(X_test, y_test),
        }
    )

comparison = pd.DataFrame(rows).round(4)
print(comparison.to_string(index=False))
```

Expected output:

```text
                 model   train    test
                 dummy  0.6268  0.6294
              logistic  0.9883  0.9860
                  tree  1.0000  0.9231
         random forest  1.0000  0.9580
     gradient boosting  1.0000  0.9580
hist gradient boosting  1.0000  0.9650
```

The ensembles beat the single tree, but the plain logistic regression is still the
best model on this split. That is an honest and important result. Breast cancer is
a small, clean, largely linearly separable dataset; ensembles earn their reputation
on messier data with interactions. A more complex model is not automatically a
better one.

## 8. Cross-Validation Instead of One Split

One split can be lucky. Compare the three tree-based models across five folds.

```python
for name in ["tree", "random forest", "gradient boosting"]:
    scores = cross_val_score(models[name], X, y, cv=5)
    print(
        f"{name}: mean={scores.mean():.4f} "
        f"folds={np.round(scores, 3)}"
    )
```

Expected output:

```text
tree: mean=0.9173 folds=[0.912 0.904 0.93  0.956 0.885]
random forest: mean=0.9614 folds=[0.921 0.947 0.991 0.974 0.973]
gradient boosting: mean=0.9631 folds=[0.93  0.947 0.974 0.982 0.982]
```

The tree's folds swing from 0.885 to 0.956, a range of seven points. The ensembles
are both higher on average and more consistent. Stability is a real benefit, not
just a higher mean. A model that scores 96% on average but varies by ten points is
harder to trust than one that scores 95% every time.

## 9. Tuning the Forest Size

`n_estimators` controls how many trees vote. More trees reduce variance but cost
time and memory.

```python
for n_trees in [10, 50, 100, 300]:
    model = RandomForestClassifier(
        n_estimators=n_trees, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(f"trees={n_trees}: test={model.score(X_test, y_test):.4f}")
```

Expected output:

```text
trees=10: test=0.9510
trees=50: test=0.9510
trees=100: test=0.9580
trees=300: test=0.9580
```

The curve flattens quickly. In practice, 100 to 500 trees is a sensible range for
datasets of this size. Beyond that, the gain is usually negligible.

## 10. Learning Rate and Boosting Rounds

Boosting has two strongly coupled knobs. First fix the number of rounds and vary
the learning rate.

```python
for learning_rate in [0.01, 0.05, 0.1, 0.5]:
    model = GradientBoostingClassifier(
        learning_rate=learning_rate,
        n_estimators=200,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    print(
        f"lr={learning_rate}: train={model.score(X_train, y_train):.4f} "
        f"test={model.score(X_test, y_test):.4f}"
    )
```

Expected output:

```text
lr=0.01: train=0.9930 test=0.9371
lr=0.05: train=1.0000 test=0.9580
lr=0.1: train=1.0000 test=0.9580
lr=0.5: train=1.0000 test=0.9650
```

A very small learning rate with only 200 rounds underfits; the model has not had
enough total correction. Moderate values are usually safest. A large learning rate
can work on an easy dataset but is riskier on noisy data, because each tree makes
big, confident steps.

Now fix the learning rate and vary the number of rounds:

```python
for n_rounds in [10, 50, 100, 200]:
    model = GradientBoostingClassifier(
        n_estimators=n_rounds, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    print(
        f"rounds={n_rounds}: train={model.score(X_train, y_train):.4f} "
        f"test={model.score(X_test, y_test):.4f}"
    )
```

Expected output:

```text
rounds=10: train=0.9883 test=0.9371
rounds=50: train=1.0000 test=0.9441
rounds=100: train=1.0000 test=0.9580
rounds=200: train=1.0000 test=0.9580
```

More rounds improve the result until the curve flattens. In noisier data the test
score would eventually turn down while training stays perfect; that is the point to
stop, and why modern libraries use early stopping.

## 11. Feature Importance

Both forest and boosting expose impurity-based importance. Compare their top
features.

```python
rf_importance = pd.Series(
    forest.feature_importances_, index=cancer.feature_names
).sort_values(ascending=False)

gb_importance = pd.Series(
    boosting.feature_importances_, index=cancer.feature_names
).sort_values(ascending=False)

print("random forest top 8")
print(rf_importance.head(8).round(4))
print("\ngradient boosting top 8")
print(gb_importance.head(8).round(4))
```

Expected output:

```text
random forest top 8
worst perimeter         0.1457
worst area              0.1441
worst concave points    0.1146
mean concave points     0.0983
worst radius            0.0724
mean radius             0.0607
mean perimeter          0.0560
mean concavity          0.0452
dtype: float64

gradient boosting top 8
worst radius            0.5589
worst concave points    0.1319
worst perimeter         0.1277
texture error           0.0371
worst texture           0.0321
worst area              0.0318
mean concave points     0.0197
worst concavity         0.0145
dtype: float64
```

The two rankings agree on the big story: size and concavity features dominate. They
disagree on details. Gradient boosting concentrates most of its credit on
`worst radius`, while the forest spreads importance across correlated features.
That difference is a known property of impurity importance, not evidence that one
model is right.

### Permutation importance

Impurity importance is biased toward features with many distinct values. Permutation
importance measures how much test accuracy drops when a feature is shuffled.

```python
result = permutation_importance(
    forest,
    X_test,
    y_test,
    n_repeats=10,
    random_state=RANDOM_STATE,
)

perm_importance = pd.Series(
    result.importances_mean, index=cancer.feature_names
).sort_values(ascending=False)

print(perm_importance.head(8).round(4))
```

Expected output:

```text
worst texture           0.0049
worst perimeter         0.0035
worst smoothness        0.0014
worst concave points    0.0007
mean concave points     0.0007
worst concavity         0.0007
mean compactness        0.0000
mean concavity          0.0000
dtype: float64
```

The absolute values are tiny because the forest is already above 95% and many
features are correlated; shuffling one still leaves the others to carry the
prediction. Treat the ranking as the signal, not the magnitude. On a harder dataset
and with a weaker model, the drops would be larger and easier to read.

## 12. Voting and Stacking

Two ways to combine *different* model families:

```python
voter = VotingClassifier(
    estimators=[
        ("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        ("forest", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
        ("boosting", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ],
    voting="soft",
)
voter.fit(X_train, y_train)

stacker = StackingClassifier(
    estimators=[
        ("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        ("forest", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
)
stacker.fit(X_train, y_train)

print("voting:", round(voter.score(X_train, y_train), 4), round(voter.score(X_test, y_test), 4))
print("stacking:", round(stacker.score(X_train, y_train), 4), round(stacker.score(X_test, y_test), 4))
```

Expected output:

```text
voting: 1.0 0.958
stacking: 0.993 0.979
```

Stacking improves on its forest member, reaching 97.9%, but neither combination
beats the simple logistic regression at 98.6%. Stacking also adds cross-validation
work and complexity. On small, clean data, combination methods mostly reproduce
what the strongest member already does. Their value appears when the members are
both strong and genuinely different.

## 13. Try It Yourself

1. **Break the forest.** Set `max_features=1` and `max_depth=3`. Does test accuracy
   fall? Explain using the bias-variance trade-off.
2. **Overfit boosting.** Use `learning_rate=1.0` and `n_estimators=1000` on a noisy
   synthetic dataset. Plot train and test accuracy as rounds increase.
3. **Compare importances.** Run permutation importance with `scoring="roc_auc"` and
   compare the ranking with impurity importance.
4. **Tune with search.** Use `GridSearchCV` to tune `n_estimators`, `max_depth`, and
   `learning_rate` for gradient boosting. How much does it improve over default?
5. **Add a weak model.** Add a KNN classifier to the voting ensemble. Does diversity
   help or hurt?
6. **Measure stability.** Repeat cross-validation with three different seeds and
   report the spread of the mean scores for the tree and the forest.

## 14. What We Learned

- Ensembles combine diverse models; averaging cancels part of their independent
  errors.
- Bagging (random forests) reduces variance; boosting reduces bias.
- A random forest turned a 92.3% tree into a 95.8% model with no tuning.
- OOB scores give a free estimate of generalization, close to the held-out score.
- Gradient boosting and its histogram variant matched or beat the forest on this
  data.
- Cross-validation showed the ensembles were not only more accurate but more stable
  than a single tree.
- `learning_rate` and `n_estimators` interact; too few rounds underfit and too many
  can overfit.
- Impurity and permutation importance tell related but different stories.
- On this small, clean dataset, logistic regression still won. Complexity must earn
  its place.

## Outcome

Learners can train random forests and gradient boosting, interpret OOB scores and
cross-validation folds, tune the main ensemble hyperparameters, compare feature
importance methods, combine models with voting and stacking, and decide when an
ensemble is worth its cost relative to a simple baseline.
