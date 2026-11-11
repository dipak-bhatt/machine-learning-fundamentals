# Day 19 — Theory: ROC Curves and Cross-Validation

**Module:** Module 5: Evaluation and Optimization

Day 18 evaluated a model at one threshold on one train/test split. Both choices hide
information. A single threshold hides how the model behaves across the full range of
operating points, and a single split hides how much the score depends on which rows
happened to land in test.

This day addresses both. **ROC curves** summarize performance across every threshold,
and **cross-validation** averages performance across several splits. Together they
give a more stable and more honest picture of a classifier.

## In Plain English

**Think of it like this:** ROC/AUC compares a model across every possible
threshold, and cross-validation tests it on several different splits instead of
one lucky split.

- The **ROC curve** plots true-positive rate against false-positive rate as the
  threshold moves.
- **AUC** is the chance a random positive scores above a random negative; 0.5 is
  random guessing.
- **Precision-recall curves** are more informative when positives are rare.
- **Cross-validation** rotates the validation set so every row is tested once.
- Use **stratified** or **grouped** splits to keep the folds fair.

## 1. Scores, Thresholds, and Labels

Most classifiers produce a score, such as the predicted probability of the positive
class. A threshold converts the score into a label:

```text
predict positive if score >= threshold, else negative
```

Precision, recall, and accuracy all depend on that threshold. Change the threshold and
the numbers change, even though the model has not changed. A threshold of 0.5 is a
convention, not a property of the model.

If we report only "recall = 0.80", a reader cannot tell whether that came from a good
model or from a very low threshold that flags almost everything. The ROC and
precision-recall curves solve this by showing all thresholds at once.

## 2. The ROC Curve

The **receiver operating characteristic (ROC)** curve plots:

- **True positive rate (TPR)** on the vertical axis. This is recall:
  `TP / (TP + FN)`.
- **False positive rate (FPR)** on the horizontal axis. This is `FP / (FP + TN)`,
  which equals `1 - specificity`.

Each point on the curve corresponds to one threshold. To build it:

1. Sort the test examples by score, descending.
2. Start at threshold above the highest score: nothing is predicted positive, so
   TPR = FPR = 0.
3. Lower the threshold one example at a time. Each newly positive example is either a
   TP (TPR rises) or an FP (FPR rises).
4. Plot the path from (0, 0) to (1, 1).

```text
TPR
1.0 |        ______
    |      _/
    |    _/        better models bow toward
    |  _/          the top-left corner
    | /
0.0 +-------------- FPR
    0.0          1.0
```

Key reference points:

- **(0, 0)**: predict nothing positive.
- **(1, 1)**: predict everything positive.
- **(0, 1)**: perfect separation, achieved only by a perfect classifier.
- The diagonal from (0, 0) to (1, 1) is random guessing.

A curve closer to the top-left corner is better. A curve below the diagonal is worse
than random, which usually means the scores are inverted.

## 3. Area Under the ROC Curve (AUC)

**AUC** is the area under the ROC curve. Its values run from 0 to 1:

- **0.5**: random ranking.
- **1.0**: perfect ranking.
- **< 0.5**: systematically inverted.

AUC has a clean probabilistic interpretation: it is the probability that a randomly
chosen positive example receives a higher score than a randomly chosen negative
example. This is a **ranking** statement, not a calibration statement. A model with
AUC 0.9 ranks well; it does not necessarily output trustworthy probabilities.

Two models can have the same AUC with different curve shapes. AUC compresses the whole
curve into one number, so it is useful for comparison but loses the operating-point
detail. Pair AUC with a curve or a chosen threshold.

## 4. Why ROC Can Mislead Under Imbalance

FPR is `FP / (FP + TN)`. When negatives vastly outnumber positives, even many false
positives produce a small FPR because the denominator is large. The ROC curve can
therefore look excellent while precision is poor.

Example: 10,000 negatives and 100 positives. A model with 200 false positives and 80
true positives has:

```text
FPR = 200 / 10,200 = 0.0196
TPR = 80 / 100 = 0.80
```

The ROC point (0.02, 0.80) looks strong. But precision is:

```text
precision = 80 / (80 + 200) = 0.286
```

Only 29% of alerts are real. For tasks where false alarms are expensive, the ROC
picture is too generous. That is why the precision-recall curve exists.

## 5. The Precision-Recall Curve

The **precision-recall (PR) curve** plots precision on the vertical axis against
recall on the horizontal axis across thresholds.

- At low recall, the model can be very precise (flag only the clearest cases).
- As recall rises, precision usually falls.
- The curve ends at the recall achieved by flagging everything.

**Average precision (AP)** summarizes the PR curve. It is the weighted mean of
precision at each recall level, weighted by the increase in recall. AP is closely
related to the area under the PR curve, but is the preferred scikit-learn summary.

```text
precision
1.0 |*
    |  *
    |    * *
    |        * * *
0.0 +-------------- recall
    0.0          1.0
```

For imbalanced data, PR is often the more informative view because precision responds
directly to false positives, while FPR is diluted by the large negative class.

### ROC versus PR

| Aspect | ROC | PR |
|--------|-----|-----|
| Axes | TPR vs FPR | precision vs recall |
| Summary | AUC | average precision |
| Baseline | 0.5 | positive rate |
| Imbalance | can look optimistic | more sensitive to false alarms |
| Use when | classes balanced, ranking quality | positives rare, precision matters |

Neither is universally better. Report the one that matches the decision. When
positives are rare and false alarms cost money, PR is usually the honest choice.

## 6. Why a Single Train/Test Split Is Not Enough

A holdout score depends on which rows were held out. With a small test set, the
estimate is noisy: a lucky or unlucky split can move accuracy by several points.

**Cross-validation (CV)** reduces this dependence by rotating which rows are held out.
The standard procedure, **k-fold CV**:

1. Split the data into `k` folds of roughly equal size.
2. For each fold `i`, train on the other `k-1` folds and evaluate on fold `i`.
3. Average the `k` scores.

```text
fold 1: [test][train][train][train][train]
fold 2: [train][test][train][train][train]
fold 3: [train][train][test][train][train]
fold 4: [train][train][train][test][train]
fold 5: [train][train][train][train][test]
```

Every row is used for testing exactly once and for training `k-1` times. The result is
`k` scores, whose mean is the estimate and whose spread indicates stability.

Common values are `k = 5` or `k = 10`. Larger `k` uses more training data per fold but
costs more computation. `k = 5` is a good default.

## 7. Stratified K-Fold

Random folds can produce training or test sets with very different class proportions,
especially when a class is rare. **Stratified k-fold** preserves the class
distribution in every fold.

For a dataset with 10% positives and `k = 5`, each fold contains about 10% positives
by construction. This reduces variance across folds and prevents a fold from having
no positives at all.

Use stratified folds for classification whenever possible. Scikit-learn's
`cross_val_score` does this automatically for classifiers that expose
`predict_proba` or `decision_function`, but being explicit with `StratifiedKFold` is
clearer and gives control over shuffling.

## 8. Other CV Variants

- **Leave-one-out (LOO)**: `k = n`. Each model trains on all but one row. Very
  expensive, low bias, high variance, rarely worth it outside tiny datasets.
- **Repeated k-fold**: run k-fold several times with different shuffles. Gives a more
  stable estimate and a better sense of variability.
- **Group k-fold**: keep all rows from the same group (patient, user, store) in the
  same fold. Prevents the same entity appearing in train and test.
- **Time-series split**: always train on the past and test on the future. Never
  shuffle time-ordered data.
- **Nested CV**: an outer loop estimates performance and an inner loop tunes
  hyperparameters, so tuning does not contaminate the estimate. Day 20 uses it.

## 9. Reading CV Results

Report the mean and the spread:

```text
CV accuracy: 0.981 +/- 0.007
```

The standard deviation across folds is a rough measure of stability, not a formal
confidence interval. With `k = 5`, it is based on only five numbers, so treat it as a
sanity check.

The **standard error** of the mean is approximately `std / sqrt(k)`. If two models
differ by less than about two standard errors, the difference may be noise. This is
why comparing single scores without variation is dangerous.

CV estimates also assume the folds are exchangeable. For grouped or time-ordered data,
that assumption fails, and standard k-fold gives optimistic scores.

## 10. Leakage in Cross-Validation

Leakage happens when information from the test fold influences training. CV makes
leakage easy to introduce accidentally, because the loop is hidden inside a library
call.

The classic mistake is preprocessing before CV:

```python
# WRONG: preprocessing sees the test folds
X_scaled = StandardScaler().fit_transform(X)
X_selected = SelectKBest(k=20).fit_transform(X_scaled, y)
scores = cross_val_score(model, X_selected, y, cv=5)
```

Every fold's test data influenced the scaler, the feature selection, and any other
transformation. With 2,000 features and a small signal, this can inflate accuracy from
about 0.50 to 0.78.

The correct pattern is a pipeline:

```python
# RIGHT: each fold fits its own preprocessing
model = make_pipeline(
    StandardScaler(),
    SelectKBest(k=20),
    LogisticRegression(),
)
scores = cross_val_score(model, X, y, cv=5)
```

The pipeline fits the scaler and selector on the training part of each fold, then
applies them to the test part. No information crosses the boundary.

The same rule applies to target encoding, imputation, PCA, resampling, and any other
step that learns parameters from data. If it calls `fit`, it belongs inside the
pipeline.

## 11. Common Mistakes

- **Reporting a single threshold as if it were the model.** Show the curve or state
  the threshold.
- **Using accuracy of a single split.** It is noisy.
- **Comparing models by one CV number.** Include the spread; prefer repeated CV.
- **Not stratifying imbalanced folds.** Some folds may contain almost no positives.
- **Preprocessing outside CV.** This inflates scores through leakage.
- **Tuning on the test set.** Use nested CV or a held-out set.
- **Shuffling time series.** It leaks the future into the past.
- **Ignoring groups.** The same patient in train and test inflates performance.
- **Assuming AUC means calibrated probabilities.** AUC is about ranking.
- **Choosing ROC when positives are rare and false alarms are costly.** Use PR.

## 12. Summary

- ROC plots TPR against FPR across all thresholds.
- AUC is the probability that a positive is ranked above a negative.
- ROC can look optimistic under strong class imbalance.
- The PR curve and average precision focus on the positive class.
- Cross-validation averages over multiple splits for a more stable estimate.
- Stratified folds preserve class proportions.
- Group and time-series splits respect data structure.
- CV variance signals stability; small differences may be noise.
- All preprocessing that learns from data must be fit inside each fold.
- Nested CV separates tuning from evaluation.

## Outcome

Learners can construct and interpret ROC and precision-recall curves, explain AUC and
average precision, choose the right curve for imbalanced data, run k-fold and
stratified cross-validation, report mean and spread, recognize grouped and time-series
requirements, avoid leakage in CV, and explain why nested CV is needed when tuning.
