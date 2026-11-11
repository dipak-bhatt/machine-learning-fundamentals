# Day 11 — Theory: Ensemble Learning

**Module:** Module 2: Supervised Learning

A single decision tree is interpretable but unstable: change a few rows and the
whole structure can change. A single line is stable but rigid. What if we trained
many models and let them vote? That is the idea behind **ensemble learning**, and it
is the workhorse of modern tabular machine learning. Random forests, gradient
boosting, and the systems that win Kaggle competitions are all ensembles.

The surprising result is that combining models can be better than the best model in
the group, even when each member is only moderately good. The reason is not magic;
it is the statistics of errors. If the members make *different* mistakes, averaging
cancels some of them. The whole art is producing that diversity and combining it
wisely.

## In Plain English

**Think of it like this:** ask many different experts and take a vote — if
their mistakes differ, the crowd is wiser than any single member.

- **Bagging** (random forest) trains many models on resampled data to reduce
  variance.
- **Boosting** trains models in sequence, each fixing the previous one's
  mistakes, to reduce bias.
- **Diversity** is the key: identical models add nothing.
- Random forests are a strong, low-effort default for tabular data.
- Gradient boosting often wins competitions but needs careful tuning.

## 1. Why Combining Models Works

Suppose each model has an error. If all models make the same error, averaging does
nothing. If their errors are independent, averaging reduces the variance. The
formal version is the **bias-variance decomposition**:

```text
expected error = bias^2 + variance + irreducible noise
```

- **Bias** is error from a model that is too simple.
- **Variance** is error from a model that is too sensitive to the training sample.
- **Noise** is the part no model can remove.

Bagging attacks variance. Boosting attacks bias. Both need **diversity**: members
that are not identical.

A useful analogy is asking many slightly different experts. If they all read the
same textbook and reason the same way, their consensus adds little. If they have
different experience and make different mistakes, the majority is more reliable
than any one of them.

## 2. Bagging: Bootstrap Aggregating

**Bagging** (bootstrap aggregating) builds many models on resampled datasets:

1. Draw `B` bootstrap samples, each of size `n`, by sampling training rows **with
   replacement**.
2. Train one model per sample.
3. Combine predictions by majority vote (classification) or averaging (regression).

Each bootstrap sample contains about 63% of the original distinct rows, because
sampling with replacement leaves roughly `1 - 1/e` unique. Different samples lead
to different models, which is the source of diversity.

Bagging is most effective with **high-variance, low-bias** learners such as deep
decision trees. A tree that overfits each sample heavily will produce very different
structures, and averaging them smooths the result. Bagging a stable linear model
does little, because all the bootstrapped models are nearly the same.

The aggregation reduces variance without increasing bias much, which is why bagged
trees generalize better than a single tree.

## 3. Random Forests

A **random forest** is bagging with one crucial addition: at every split, the tree
may consider only a random subset of the features. Instead of searching all `p`
features, it searches `max_features` of them.

This feature randomness does two things:

- It **decorrelates** the trees. Without it, a few strong features would dominate
  every tree, and all trees would look alike.
- It **forces diversity**, even when one feature is much more informative than the
  rest.

At prediction time, every tree votes and the forest returns the majority class or
the average probability. Individual trees are deep and overfit, but the forest as a
whole is smooth.

### Out-of-bag error

Because each bootstrap sample omits about a third of the rows, those omitted rows
act as a free validation set for that tree. Predicting them with the trees that did
not see them gives the **out-of-bag (OOB) score**. It is a useful estimate of
generalization that requires no separate holdout, though it is not a substitute for
a properly held-out test set.

### Hyperparameters

| Parameter | Effect |
|-----------|--------|
| `n_estimators` | number of trees; more is usually better until returns flatten |
| `max_features` | features considered per split; lower means more diversity |
| `max_depth` | tree depth; deep trees fit more but cost more |
| `min_samples_leaf` | minimum rows per leaf; larger values smooth the forest |
| `oob_score` | compute the out-of-bag estimate |
| `n_jobs` | parallel cores; forests train embarrassingly in parallel |

Random forests are robust to hyperparameter choices, need no scaling, and rarely
overfit as you add trees. They are an excellent default for tabular data.

## 4. Boosting: Learning from Mistakes

**Boosting** builds models sequentially, each one focused on the errors of the
previous ones. The final prediction is a weighted sum of all members.

The two classic families are:

- **AdaBoost** reweights training rows. After each round, misclassified rows get
  larger weights, so the next weak learner focuses on them.
- **Gradient boosting** fits each new model to the **residuals** (or, more
  generally, the negative gradient of the loss) of the current ensemble.

Boosting primarily reduces **bias**. A single shallow tree (a "stump") is weak, but
hundreds of stumps in sequence can approximate a complex function. Unlike bagging,
boosting is sequential: each model depends on the previous ones, so it cannot be
parallelized across trees in the same way.

## 5. Gradient Boosting in Detail

Gradient boosting treats the ensemble as a function `F(x)` and builds it additively:

```text
F_0(x) = constant
F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)
```

At step `m`, the new tree `h_m` is fit to the negative gradient of the loss with
respect to the current predictions. For squared error, the negative gradient is
simply the residual `y - F_{m-1}(x)`, so each tree fits what the ensemble still
gets wrong. For log loss, it fits a related quantity that pushes probabilities in
the right direction.

Key controls:

| Parameter | Effect |
|-----------|--------|
| `n_estimators` | number of boosting rounds; too many can overfit |
| `learning_rate` | shrink factor per tree; smaller needs more trees |
| `max_depth` | depth of each weak tree; usually 3-5 for tabular data |
| `subsample` | fraction of rows per round; below 1 adds randomness |
| `min_samples_leaf` | smooths individual trees |

`learning_rate` and `n_estimators` trade off directly. A small learning rate with
many trees often generalizes better but trains more slowly. This is why boosting
requires more tuning than a random forest.

### The overfitting story

Bagging usually improves with more trees. Boosting can overfit if you keep adding
trees, because later trees start fitting noise. Validation curves and early
stopping are essential. Modern libraries stop when validation loss stops improving.

## 6. XGBoost, LightGBM, and CatBoost

The three dominant gradient-boosting libraries are optimized implementations with
extra features:

- **XGBoost** adds regularization terms, handles missing values, and uses efficient
  split finding. It popularized gradient boosting in industry.
- **LightGBM** grows trees leaf-wise and uses histogram binning, making it very fast
  on large data.
- **CatBoost** handles categorical features natively and reduces a subtle form of
  target leakage during encoding.

They share the same core idea but differ in speed, defaults, and categorical
handling. scikit-learn's `GradientBoostingClassifier` and
`HistGradientBoostingClassifier` are good educational implementations and often
sufficient in practice.

## 7. Voting and Stacking

Two other combination strategies are worth knowing:

**Voting** trains several different models and combines their predictions. Hard
voting takes the majority class; soft voting averages predicted probabilities.
Voting works best when the base models are strong and make different kinds of
errors.

**Stacking** trains a **meta-model** on the predictions of several base models. The
base models are trained with cross-validation so the meta-model sees out-of-fold
predictions, reducing leakage. Stacking can outperform simple voting but is more
complex and easier to get wrong.

Neither is a substitute for good base models. Stacking noisy models tends to stack
the noise.

## 8. Strengths and Weaknesses

**Random forests**

- Strong default accuracy with little tuning.
- Robust to outliers and irrelevant features.
- No feature scaling required.
- Provide OOB estimates and feature importance.
- Slower to predict than one tree; larger models.
- Less interpretable than a single tree; importance is still available.

**Gradient boosting**

- Often the best accuracy on tabular data.
- Handles complex nonlinearities and interactions.
- Regularization and early stopping control overfitting.
- More hyperparameters and longer tuning.
- Sequential training is slower than a forest.
- Easy to overfit with too many rounds or too high a learning rate.

**Both**

- Need honest validation; training scores are meaningless.
- Feature importance is useful but biased and not causal.
- Can encode and amplify biases present in the data.

## 9. Interpretability

An ensemble is not a flowchart. You cannot read one path and understand the whole
prediction. Still, several tools help:

- **Feature importance** ranks which inputs the ensemble relies on. Impurity-based
  importance is fast but biased; permutation importance is slower but more honest.
- **Partial dependence plots** show the average effect of one feature while
  marginalizing the others.
- **SHAP values** attribute a single prediction to individual features, giving
  local explanations.
- **Surrogate models** fit a shallow tree to the ensemble's predictions to recover
  an approximate rule set.

In regulated settings, the loss of transparency may outweigh the accuracy gain. The
right choice depends on the application, not only the score.

## 10. Real-World Use Cases

Ensembles dominate tabular applications:

- **Fraud detection:** gradient boosting combines transaction amount, time, device,
  and history. The class imbalance makes precision-recall evaluation essential.
- **Churn prediction:** random forests and boosting rank customers by risk; feature
  importance guides retention campaigns.
- **Credit scoring:** boosted trees capture interactions among income, debt, and
  payment history, often blended with logistic regression for explainability.
- **Demand and business forecasting:** gradient boosting on calendar, price, and
  promotion features predicts sales at the store-day level.

In each case the ensemble is paired with careful validation, monitoring, and a
business-aware metric. A high AUC on a stale test set is not a deployed success.

## 11. Common Mistakes

- **Evaluating on training data.** Forests and boosting reach perfect training
  accuracy; that tells you nothing.
- **Tuning on the test set.** Use cross-validation and touch the test set once.
- **Assuming more trees always help boosting.** More rounds can overfit; use early
  stopping.
- **Ignoring the learning rate.** It interacts strongly with `n_estimators`.
- **Trusting impurity importance alone.** It favors high-cardinality features; check
  permutation importance.
- **Forgetting class imbalance.** Use stratified splits and metrics such as recall,
  F1, or PR-AUC.
- **Believing the ensemble is unbiased.** It learns whatever patterns, including
  discriminatory ones, are in the data.
- **Skipping the simple baseline.** A regularized logistic model sometimes wins on
  small, clean datasets.

## 12. Summary

- Ensembles combine diverse models; diversity is what makes averaging useful.
- Bagging trains on bootstrap samples and reduces variance.
- Random forests add feature randomness and are a strong tabular default.
- Boosting builds models sequentially on residuals and reduces bias.
- Gradient boosting is the accuracy workhorse, at the cost of more tuning.
- `learning_rate`, `n_estimators`, and tree depth are the key boosting knobs.
- Voting and stacking combine different model families.
- Feature importance and SHAP help interpret ensembles, but transparency is always
  reduced compared with a single tree.

## Outcome

Learners can explain why ensembles improve on individual models, describe bagging
versus boosting, interpret random forest and gradient boosting hyperparameters,
compute and use out-of-bag error, compare ensembles against a single tree and a
baseline, and discuss the accuracy-interpretability trade-off in real applications.
