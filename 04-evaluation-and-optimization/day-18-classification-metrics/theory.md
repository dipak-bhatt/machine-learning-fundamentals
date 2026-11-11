# Day 18 — Theory: Classification Metrics

**Module:** Module 5: Evaluation and Optimization

A model is only as useful as the way we measure it. Accuracy, the fraction of
predictions that are correct, is intuitive and often misleading. On a dataset where
95% of cases are negative, a model that always predicts "negative" is 95% accurate
and completely useless. Choosing the right metric is part of the modeling problem,
not an afterthought.

This day builds the vocabulary of classification evaluation: the confusion matrix,
precision, recall, specificity, and F1. It explains when each one is appropriate, how
to handle multiple classes, and why the same model can look excellent or terrible
depending on the metric and the threshold.

## In Plain English

**Think of it like this:** accuracy is one overall grade; precision, recall, and
F1 tell you *which kind* of mistakes you are making.

- The **confusion matrix** counts true/false positives and negatives.
- **Precision** = of the cases we flagged, how many were right?
- **Recall** = of the real cases, how many did we find?
- **F1** balances precision and recall in one number.
- Accuracy misleads on imbalanced data; pick the metric that matches the cost of
  each mistake.

## 1. The Setting

In binary classification each prediction falls into one of four cells:

| | predicted positive | predicted negative |
|---|---|---|
| **actual positive** | true positive (TP) | false negative (FN) |
| **actual negative** | false positive (FP) | true negative (TN) |

- **TP**: correctly flagged positives.
- **TN**: correctly left as negative.
- **FP**: false alarm; predicted positive but actually negative.
- **FN**: missed case; predicted negative but actually positive.

Everything below is a ratio of these four numbers. The **confusion matrix** is the
table itself, and it is the most informative single summary of a classifier because
it shows the kinds of errors, not just their count.

## 2. Accuracy

```text
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Accuracy answers: of all predictions, how many were right? It is easy to explain and
appropriate when classes are balanced and all errors cost roughly the same.

Accuracy fails under **class imbalance**. If 2% of transactions are fraudulent, a
model that predicts "not fraud" for every transaction is 98% accurate while catching
no fraud at all. Accuracy also hides which error is being made. A spam filter that
blocks legitimate mail and one that lets spam through can share the same accuracy.

## 3. Precision

```text
precision = TP / (TP + FP)
```

Precision answers: when the model says positive, how often is it right? It measures
the purity of positive predictions.

High precision matters when a false positive is expensive:

- A spam filter that deletes a legitimate invoice.
- A fraud alert that freezes an innocent customer's card.
- A medical test that leads to an invasive follow-up.

Precision ignores false negatives entirely. A model that flags one very likely
positive and ignores everything else can have perfect precision while missing most
cases.

## 4. Recall (Sensitivity)

```text
recall = TP / (TP + FN)
```

Recall answers: of all the actual positives, how many did the model find? It measures
coverage.

High recall matters when a false negative is expensive:

- Missing a cancer diagnosis.
- Failing to detect fraud.
- A safety system that misses a hazardous condition.

Recall ignores false positives. A model that flags everything has perfect recall and
terrible precision.

## 5. Specificity

```text
specificity = TN / (TN + FP)
```

Specificity answers: of all the actual negatives, how many were correctly left
alone? It is the mirror image of recall, computed on the negative class.

Recall and specificity trade off directly. Lowering the threshold flags more cases:
recall rises, specificity falls. Reporting both makes the trade-off explicit. The
combination is sometimes summarized as **balanced accuracy**:

```text
balanced accuracy = (recall + specificity) / 2
```

Balanced accuracy gives each class equal weight, so it does not reward ignoring the
minority class.

## 6. F1 Score

Precision and recall pull in opposite directions. The **F1 score** combines them with
a harmonic mean:

```text
F1 = 2 * (precision * recall) / (precision + recall)
```

The harmonic mean punishes imbalance: if either precision or recall is low, F1 is
low. A model with precision 1.0 and recall 0.1 has F1 ≈ 0.18, correctly signalling
that it is not useful.

F1 assumes precision and recall are equally important. That is often convenient but
not always true. The **F-beta score** generalizes F1:

```text
F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
```

- `beta < 1` weights precision more.
- `beta > 1` weights recall more.

If missing a positive is twice as costly as a false alarm, encode that with `beta`.

## 7. Multi-Class Classification

With more than two classes, the confusion matrix is `n_classes x n_classes`. Each row
is the actual class; each column is the predicted class. The diagonal holds correct
predictions; off-diagonal cells show which classes are confused.

Per-class precision and recall are computed by treating one class as positive and
all others as negative (**one-vs-rest**):

```text
precision_class_i = TP_i / (TP_i + FP_i)
recall_class_i    = TP_i / (TP_i + FN_i)
```

There are several ways to reduce the per-class scores to one number.

### Macro average

Compute the metric for each class, then take the unweighted mean.

```text
macro_precision = mean(precision_0, precision_1, ..., precision_n)
```

Every class counts equally, so rare classes matter as much as common ones. Use macro
averaging when you care about performance on every class, including small ones.

### Weighted average

Compute the metric for each class, then average weighted by class support (the number
of true examples).

```text
weighted_precision = sum(support_i * precision_i) / sum(support_i)
```

This reflects overall performance on the dataset as it is distributed. A large class
dominates. Use it when you care about the aggregate, not each class equally.

### Micro average

Pool the counts across classes, then compute the metric once.

```text
micro_precision = sum(TP_i) / (sum(TP_i) + sum(FP_i))
```

For single-label multi-class classification, micro precision, recall, and F1 all
equal accuracy, because every prediction belongs to exactly one class. Micro
averaging becomes more interesting in multi-label settings, where an example can
have several true labels.

| Average | Treats classes | Good when |
|---------|----------------|-----------|
| macro | equally | rare classes matter |
| weighted | by frequency | overall dataset performance |
| micro | pooled counts | multi-label / aggregate counts |

Always state which average you used. "F1 = 0.80" is incomplete without it.

## 8. Choosing the Right Metric

Start from the cost of errors, not from the metric you know best.

| Task | False negative cost | False positive cost | Metric focus |
|------|---------------------|---------------------|--------------|
| Cancer screening | very high | moderate | recall, then precision |
| Spam filter | low | high | precision, then recall |
| Fraud detection | high | moderate | recall with acceptable precision |
| Search ranking | low | moderate | precision at k |
| Balanced benchmark | similar | similar | accuracy or F1 |

A **cost matrix** makes this explicit. Assign a cost to each cell of the confusion
matrix and choose the model or threshold that minimizes expected cost:

```text
expected_cost = c_FN * FN + c_FP * FP + c_TP * TP + c_TN * TN
```

Often `c_TP = c_TN = 0`, so the decision reduces to the ratio of `c_FN` to `c_FP`.
This ratio should drive the threshold, not habit.

## 9. Class Imbalance and the Accuracy Paradox

The **accuracy paradox** is the pattern where a trivial model beats a real one on
accuracy while being useless. Two common variants:

- Always predicting the majority class gives high accuracy when the minority is
  small.
- A model tuned for accuracy learns to ignore the minority because that minimizes
  errors overall.

Ways to address imbalance:

- **Choose a better metric**: recall, F1, balanced accuracy, or precision at a fixed
  recall.
- **Resample**: oversample the minority, undersample the majority, or generate
  synthetic examples (SMOTE).
- **Class weights**: tell the model that minority errors cost more
  (`class_weight="balanced"`).
- **Adjust the threshold**: lower it to catch more positives.
- **Collect more data**: often the most effective fix for a genuinely rare class.

Resampling changes the class distribution, which changes predicted probabilities.
Recalibrate or re-tune the threshold after resampling.

## 10. The Precision-Recall Trade-off

Most classifiers output a score, and the threshold converts the score into a label.
The default threshold of 0.5 is a convention, not a law.

Raising the threshold:

- Fewer positive predictions.
- Precision tends to rise.
- Recall tends to fall.

Lowering the threshold does the reverse: more positives, higher recall, lower
precision. The full curve of (recall, precision) across thresholds is the
**precision-recall curve**.

A model can have several operating points, each suited to a different cost regime. In
practice, choose the threshold from the business or clinical requirement:

- "Flag enough cases to catch 90% of fraud" fixes recall, then read precision.
- "Keep false alarms under 5%" fixes precision, then read recall.

Reporting a single threshold without saying why is a hidden assumption.

## 11. A Note on ROC and AUC

The **receiver operating characteristic (ROC)** curve plots true positive rate
(recall) against false positive rate (`1 - specificity`) across thresholds. The area
under the curve (AUC) summarizes ranking quality: the probability that a random
positive is scored above a random negative.

ROC/AUC is threshold-independent and useful for comparing models. But under strong
imbalance, a high AUC can coexist with poor precision, because the false positive
rate stays small when there are many negatives. The precision-recall curve is often
more informative there. Day 19 covers both in detail.

## 12. Reporting Best Practices

- **Report the confusion matrix**, or at least TP/FP/FN/TN.
- **State the metric and the averaging** (macro, weighted, micro).
- **Report precision and recall together**, not one alone.
- **State the threshold** if scores were thresholded.
- **Show class distribution**, so imbalance is visible.
- **Report uncertainty**, such as confidence intervals or fold-to-fold variation.
- **Compare against a baseline**, such as the majority-class predictor.
- **Avoid precision on tiny denominators** without a caveat.

Metric choices should be decided before looking at test results, otherwise it is easy
to pick whichever number looks best.

## 13. Common Mistakes

- **Optimizing accuracy on imbalanced data.** The majority class wins.
- **Reporting recall without precision**, or the reverse.
- **Using the default threshold without justification.** It may not match costs.
- **Forgetting to state the averaging** in multi-class reports.
- **Comparing F1 across datasets with different class distributions.** F1 is not
  comparable in the abstract.
- **Tuning on the test set.** Metric shopping leaks information.
- **Ignoring the confusion matrix.** Aggregates hide which errors occur.
- **Assuming F1 is always right.** It encodes equal precision/recall weight.
- **Evaluating after resampling without recalibration.** Probabilities shift.
- **Treating a metric as a goal.** The goal is the decision, not the number.

## 14. Summary

- The confusion matrix is the foundation; all metrics derive from it.
- Accuracy is misleading under imbalance.
- Precision measures purity of positive predictions.
- Recall measures coverage of actual positives.
- Specificity measures correct negatives; balanced accuracy combines recall and
  specificity.
- F1 is the harmonic mean of precision and recall; F-beta adjusts the weighting.
- Multi-class scores need an averaging choice: macro, weighted, or micro.
- Choose metrics from error costs, not convenience.
- The threshold is a decision variable, not a fixed 0.5.
- ROC/AUC is threshold-independent; precision-recall is often better under imbalance.

## Outcome

Learners can build and read a confusion matrix, compute precision, recall,
specificity, F1, and balanced accuracy, explain macro, weighted, and micro averages,
choose metrics from error costs, diagnose the accuracy paradox, adjust thresholds and
class weights, and report classification results honestly.
