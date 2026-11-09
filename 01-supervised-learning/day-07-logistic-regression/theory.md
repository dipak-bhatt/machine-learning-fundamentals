# Day 07 — Theory: Logistic Regression and Classification

**Module:** Module 2: Supervised Learning

On Day 05 and Day 06 we predicted a **number**: a house price. We learned a line,
measured its errors, and improved it with gradient descent. Many real questions do
not have a number as the answer. They have a **choice**: Is this email spam or not?
Is this tumor malignant or benign? Will this customer leave or stay?

Today we meet **logistic regression**, the standard first model for classification.
Despite its name, it is a classification method. It still computes a linear score,
but then turns that score into a probability between 0 and 1. We will see how
probabilities become class labels, why the loss function changes, how to read a
confusion matrix, and how the same ideas extend to more than two classes.

## 1. From Predicting Numbers to Predicting Choices

A **classification** problem asks the model to assign one of a fixed set of labels.
The labels are also called **classes**.

| Question | Target type | Task |
|----------|-------------|------|
| How much will this house sell for? | a number | regression |
| Is this email spam? | spam / not spam | binary classification |
| Which species is this flower? | setosa / versicolor / virginica | multiclass classification |
| Will this customer cancel next month? | churn / no churn | binary classification |
| Which products will this user click? | several labels at once | multilabel classification |

The first kind stays with regression. The rest are classification.

### Binary, multiclass, and multilabel

- **Binary classification** has exactly two classes. One is usually called the
  **positive** class and the other the **negative** class. The choice of which is
  which is a convention, but it matters for how we read metrics.
- **Multiclass classification** has three or more mutually exclusive classes. A
  flower is one species, not two.
- **Multilabel classification** allows several labels at once. A news article can
  be about sport *and* business. This is a later topic; today we focus on binary
  and multiclass.

### Why labels are usually encoded as 0 and 1

Computers prefer numbers, so we map the two classes to `0` and `1`. For example:

```text
benign     -> 0
malignant  -> 1
```

This encoding is arbitrary but useful: the positive class becomes `y = 1`, and a
predicted probability `p` can be read as "the probability that this row belongs to
class 1". We will use that convention throughout.

| Symbol | Meaning |
|--------|---------|
| `n` | number of rows |
| `p` | number of features |
| `x_i` | the feature values of row `i` |
| `y_i` | the true label of row `i`, either 0 or 1 |
| `z_i` | the linear score for row `i` |
| `p_i` | the predicted probability that row `i` is positive |

The same feature table from regression is used here. Nothing about the inputs
changes; only the target and the loss change.

## 2. Why Ordinary Linear Regression Is the Wrong Tool

Suppose we encode malignant as 1 and benign as 0, then fit a straight line with
linear regression. Three problems appear immediately.

**Problem 1: predictions leave the allowed range.** A line can predict `-0.4` or
`1.7`. Neither is a valid probability. We could clip the outputs to `[0, 1]`, but
clipping throws away information and creates flat regions where the model cannot
learn.

**Problem 2: the loss does not match the task.** Squared error treats a prediction
of `0.8` and a prediction of `1.2` as equally wrong (`0.04` each), even though
`1.2` is not a valid probability. Linear regression is built to estimate a
conditional mean, not a probability. It also gives no principled way to say "I am
90% sure" versus "I am 51% sure".

**Problem 3: the decision boundary is arbitrary.** If we threshold the line at
`0.5`, nothing in the fitting process was designed to make that threshold
meaningful. The line is chosen to minimize squared distance to 0 and 1, not to
separate the classes well.

We need a model that:

1. always outputs a number between 0 and 1;
2. treats that number as a probability;
3. has a loss function that rewards confident correct predictions and punishes
   confident wrong ones;
4. has a decision boundary we can understand and move.

Logistic regression is the simplest model with all four properties.

## 3. The Classification Setup

We have a table of features `X` and a column of labels `y`.

```text
row 1:  x_1 = [radius, texture, ...]   y_1 = 0
row 2:  x_2 = [radius, texture, ...]   y_2 = 1
...
row n:  x_n = [radius, texture, ...]   y_n = 1
```

The model's job is to estimate:

```text
P(y = 1 | x)
```

That is, the probability that the row is positive, given its features. Once we
have that probability, we can produce a label by comparing it with a threshold:

```text
predict 1 if P(y = 1 | x) >= threshold
predict 0 otherwise
```

The rest of the theory explains how `P(y = 1 | x)` is computed, how the parameters
are learned, and how to judge the result.

### Class balance

Before fitting anything, count the classes:

```text
number of positives / number of rows
```

If the classes are roughly equal, accuracy is a reasonable first summary. If one
class is much more common, accuracy can be misleading, as Section 11 shows. The
Breast Cancer Wisconsin dataset used in the practical has 212 malignant and 357
benign rows, a mild imbalance that already makes a "always predict benign" baseline
look 63% accurate.

## 4. A Linear Score, Then a Squash

Logistic regression starts exactly like linear regression. It computes a weighted
sum of the features plus an intercept:

```text
z = b + w1*x1 + w2*x2 + ... + wp*xp
```

Here `z` is called the **linear score**, **logit**, or **log-odds**. It can be any
real number, from very negative to very positive. A large positive `z` means the
evidence points toward the positive class; a large negative `z` means it points
toward the negative class.

The score is then passed through the **sigmoid function**:

```text
sigma(z) = 1 / (1 + exp(-z))
```

The output is always between 0 and 1:

```text
probability = sigma(z)
```

| `z` | `sigma(z)` | Reading |
|----:|-----------:|---------|
| -3 | 0.047 | strong evidence for the negative class |
| -2 | 0.119 | fairly likely negative |
| -1 | 0.269 | leaning negative |
| 0 | 0.500 | perfectly undecided |
| 1 | 0.731 | leaning positive |
| 2 | 0.881 | fairly likely positive |
| 3 | 0.953 | strong evidence for the positive class |

Two useful properties:

```text
sigma(0) = 0.5
sigma(-z) = 1 - sigma(z)
```

The second property says the sigmoid is symmetric around 0.5. A score of `-z` is
exactly as far from 0.5 on the low side as `+z` is on the high side.

### The sigmoid has a convenient derivative

For gradient descent we need the derivative. It is unusually tidy:

```text
d sigma(z) / dz = sigma(z) * (1 - sigma(z))
```

In words: the slope is largest at `z = 0` (where the probability is 0.5) and
becomes tiny far out in the tails. That is why a very confident model learns slowly
from a single example: the sigmoid is nearly flat there.

### Why not a step function?

A step function would directly output 0 or 1:

```text
predict 1 if z > 0, else 0
```

That is fine for making a decision, but it is terrible for learning. Its slope is
zero almost everywhere and undefined at the jump, so gradient descent has nothing
to follow. The sigmoid is a smooth version of the same idea.

## 5. Odds, Log-Odds, and Why the Model Is "Logistic"

A probability is not the only way to describe chance. **Odds** compare the chance
of success with the chance of failure:

```text
odds = p / (1 - p)
```

For example, `p = 0.8` means the odds are `0.8 / 0.2 = 4`, often stated as "4 to
1". The **log-odds**, also called the **logit**, is:

```text
logit(p) = log( p / (1 - p) )
```

| Probability `p` | Odds | Log-odds |
|----------------:|-----:|---------:|
| 0.01 | 0.010 | -4.595 |
| 0.10 | 0.111 | -2.197 |
| 0.50 | 1.000 | 0.000 |
| 0.80 | 4.000 | 1.386 |
| 0.90 | 9.000 | 2.197 |
| 0.99 | 99.000 | 4.595 |

The logit turns a probability in `(0, 1)` into any real number. The sigmoid does
the reverse. They are inverses of each other:

```text
z = logit(p)  <->  p = sigma(z)
```

Logistic regression assumes that the log-odds are a **linear function** of the
features:

```text
log( p / (1 - p) ) = b + w1*x1 + ... + wp*xp
```

That single equation is the whole model. It is called "logistic" because it uses
the logistic function (the sigmoid) to map the linear score back to a probability.

### What a coefficient means

Because the model is linear in the log-odds, a one-unit increase in feature `j`
adds `w_j` to the log-odds, holding the other features fixed. Exponentiating both
sides shows what happens to the odds:

```text
odds multiply by exp(w_j)
```

- If `w_j = 0`, the feature has no effect: odds multiply by 1.
- If `w_j > 0`, the feature increases the odds of the positive class.
- If `w_j < 0`, the feature decreases the odds of the positive class.

For example, a coefficient of `0.7` means a one-unit increase in that feature
multiplies the odds by `exp(0.7) ≈ 2.01`, roughly doubling them. A coefficient of
`-0.7` roughly halves them.

A caution: the effect on the **probability** is not constant. When the predicted
probability is already near 0 or 1, adding the same amount to the log-odds changes
the probability very little. The sigmoid is flat in the tails. This is why we
describe coefficients in terms of odds, not "percentage points of probability".

## 6. Decision Boundaries and Thresholds

The model outputs a probability. To produce a class label, we choose a
**threshold** `t`:

```text
predict 1 if p >= t
predict 0 if p < t
```

The default is `t = 0.5`. Because `p = 0.5` exactly when `z = 0`, the default
boundary is:

```text
b + w1*x1 + ... + wp*xp = 0
```

With one feature, this is a single point on the number line. With two features, it
is a straight line. With more features, it is a **hyperplane**. Logistic regression
is therefore a **linear classifier**: its decision boundary is always linear in the
input features.

### The boundary is not the same as the threshold

The boundary is a surface in feature space. The threshold is a number in
probability space. Changing the threshold changes where the boundary sits:

- A lower threshold makes the positive class easier to predict, so the boundary
  moves toward the negative side.
- A higher threshold makes the positive class harder to predict, so the boundary
  moves toward the positive side.

### Why 0.5 is not always the right threshold

The best threshold depends on the **cost of mistakes**:

| Situation | Costly mistake | Sensible move |
|-----------|----------------|---------------|
| Cancer screening | missing a malignant tumor (false negative) | lower the threshold, accept more false alarms |
| Spam filtering | deleting a real email (false positive) | raise the threshold, accept more spam |
| Fraud detection | missing fraud (false negative) | lower the threshold, investigate more alerts |
| Closing a factory line | stopping production for no reason (false positive) | raise the threshold |

A threshold is a **decision rule**, not a property of the model. The model's
probabilities stay the same; only the label we assign changes. The practical shows
this directly on held-out data.

## 7. Log Loss: the Right Loss for Probabilities

For regression we used squared error. For classification with probabilities we use
**log loss**, also called **binary cross-entropy**.

For one row with true label `y` (0 or 1) and predicted probability `p`:

```text
loss = -[ y * log(p) + (1 - y) * log(1 - p) ]
```

When `y = 1`, the second term disappears:

```text
loss = -log(p)
```

When `y = 0`, the first term disappears:

```text
loss = -log(1 - p)
```

The average log loss over `n` rows is:

```text
J = -(1/n) * sum( y_i * log(p_i) + (1 - y_i) * log(1 - p_i) )
```

### How the penalty behaves

| True label | Predicted `p` | Loss | Reading |
|-----------:|--------------:|-----:|---------|
| 1 | 0.99 | 0.010 | confident and correct, tiny penalty |
| 1 | 0.90 | 0.105 | correct, small penalty |
| 1 | 0.60 | 0.511 | correct but unsure, moderate penalty |
| 1 | 0.50 | 0.693 | no information, the penalty of a coin flip |
| 1 | 0.10 | 2.303 | confident and wrong, large penalty |
| 1 | 0.01 | 4.605 | very confident and wrong, huge penalty |
| 0 | 0.999 | 6.908 | very confident and wrong, enormous penalty |

This is the behavior we want. Log loss cares about **confidence**, not just whether
the label was on the correct side of 0.5. A model that says "90% malignant" and is
wrong should be punished much more than one that says "51% malignant" and is wrong.

### Why not accuracy as the training loss?

Accuracy counts a prediction as correct or incorrect and ignores the probability.
It is a step function, so it has zero gradient almost everywhere. Small changes in
the weights often do not change the number of correct predictions at all, which
makes learning impossible by gradient descent. Accuracy is a useful **reporting**
metric, but a poor optimization target.

### Why not squared error?

Squared error can be used with a sigmoid, but it interacts badly with the sigmoid's
flat tails: when the model is confidently wrong, the gradient can become very small,
so learning stalls. Log loss is designed for probabilities and gives strong
gradients exactly where they are needed.

### Baselines for log loss

A model must beat a simple baseline. For classification, the simplest baseline
predicts the **class prior**: the proportion of positives in the training data, for
every row. On the practical split, that prior is about `0.374`, and its log loss is
about **0.658**. The trained model reaches about **0.077**. Lower is better, and
0 would mean perfect, fully confident predictions.

Log loss is also affected by **calibration**: a model that predicts the right labels
but with poorly scaled probabilities can have a worse log loss than a model with
better-calibrated probabilities. We return to calibration in Day 18 and Day 19.

## 8. How Logistic Regression Learns

### Maximum likelihood

For each row, the model says "the probability of the observed label is `p_i` if the
label is 1, and `1 - p_i` if the label is 0". Multiplying these probabilities over
all rows gives the **likelihood** of the observed labels. Choosing parameters that
make the observed labels as likely as possible is called **maximum likelihood
estimation**.

Taking the logarithm turns the product into a sum, and maximizing the log
likelihood is exactly the same as minimizing log loss. So the two ideas agree:

```text
maximize likelihood  <->  minimize log loss
```

### The gradient

We need the derivative of the log loss with respect to each weight. For one row,
with `z = b + w·x`, `p = sigma(z)`, and true label `y`:

```text
dJ/dp = -y/p + (1 - y)/(1 - p)
dp/dz = p * (1 - p)
```

Multiply them using the chain rule:

```text
dJ/dz = [ -y/p + (1 - y)/(1 - p) ] * p * (1 - p)
      = -y * (1 - p) + (1 - y) * p
      = -y + y*p + p - y*p
      = p - y
```

This is remarkably simple: the derivative with respect to the score is just
`predicted probability - true label`. Then, because `z = b + w·x`:

```text
dJ/db = p - y
dJ/dw_j = (p - y) * x_j
```

Averaging over all rows:

```text
gradient_b = (1/n) * sum( p_i - y_i )
gradient_w = (1/n) * sum( (p_i - y_i) * x_i )
```

In vector form, with `X` including a column of ones and `p` the vector of predicted
probabilities:

```text
gradient = (1/n) * X^T * (p - y)
```

### Comparing with linear regression

| Model | Prediction | Gradient |
|-------|------------|----------|
| Linear regression | `y_hat = X * theta` | `(2/n) * X^T * (y_hat - y)` |
| Logistic regression | `p = sigma(X * theta)` | `(1/n) * X^T * (p - y)` |

The structure is the same: a prediction minus the truth, multiplied by the features.
The difference is the prediction: a number for regression, a probability for
classification. The factor `2` is absent for log loss because the loss does not
contain a square.

### One update by hand

Take one row with `x = 1`, `y = 1`. Start at `b = 0`, `w = 0`, and use a large
learning rate `alpha = 1.0` so the change is easy to see.

```text
z = b + w*x = 0
p = sigma(0) = 0.5
gradient_b = p - y = -0.5
gradient_w = (p - y) * x = -0.5

new b = 0 - 1.0 * (-0.5) = 0.5
new w = 0 - 1.0 * (-0.5) = 0.5
```

After the update:

```text
z = 0.5 + 0.5 * 1 = 1.0
p = sigma(1.0) = 0.731
```

The model moved from a coin flip (0.5) toward the correct label (1.0). As with
linear regression, one row is not enough to learn a useful model; the update is
repeated over many rows and iterations.

### There is no closed-form solution

Unlike linear regression, logistic regression has no simple formula like the normal
equation. The sigmoid makes the equations nonlinear in the parameters, so we use an
iterative optimizer. scikit-learn uses solvers such as `lbfgs`, `liblinear`, and
`saga`; each minimizes a log-loss objective with the chosen regularization. The
gradient formula above is the core of that process.

If scikit-learn prints a convergence warning, it means the solver stopped before
settling. Common fixes are:

- increase `max_iter`;
- standardize the features;
- reduce the regularization strength if appropriate.

### Regularization

Like Ridge and Lasso, logistic regression can penalize large weights. By default
scikit-learn applies an **L2 penalty**. The strength is controlled by `C`, which is
the **inverse** of the regularization strength:

```text
small C  -> strong penalty -> simpler, flatter model
large C  -> weak penalty   -> more flexible model
```

The default is `C = 1.0`. An L1 penalty can produce sparse coefficients, which is
useful when many features are irrelevant, but it needs a solver that supports it
(for example `liblinear` or `saga`).

As with Ridge and Lasso, **standardize features before regularizing**. Otherwise the
penalty falls unevenly on features measured in different units.

### Scaling and multicollinearity

Standardization also helps the solver converge, especially when features have very
different ranges (radius in one unit, area in another). When two features are
almost duplicates, their coefficients can become unstable: the model can increase
one and decrease the other while leaving predictions nearly unchanged. The model
may still predict well, but individual coefficients are hard to interpret.

## 9. Interpreting the Coefficients

The sign of a coefficient tells us the direction of the association with the
positive class:

- positive: larger feature values are associated with higher odds of the positive
  class;
- negative: larger feature values are associated with lower odds.

The magnitude tells us how strong that association is **on the log-odds scale**.
Because features have different units, compare coefficients only after
standardizing, or convert them to odds ratios.

### A one-feature example

Using only `mean radius` on the breast cancer data, the fitted model is:

```text
log-odds(malignant) = -13.566 + 0.913 * mean_radius
```

- A one-unit increase in `mean radius` adds `0.913` to the log-odds.
- The odds multiply by `exp(0.913) ≈ 2.49`.
- On standardized features, the coefficient is about `3.265`, so a one
  standard-deviation increase multiplies the odds by `exp(3.265) ≈ 26.2`.

The model gives:

| Mean radius | Probability malignant |
|------------:|----------------------:|
| 10 | 0.012 |
| 15 | 0.533 |
| 20 | 0.991 |

This matches the medical intuition that larger cell nuclei are more suspicious, but
remember that these are **associations in this dataset**, not proof of causation.
Radius alone reaches about 86% accuracy; all 30 features reach about 96%, so the
other measurements add real information.

### Cautions

- **Correlated features:** the breast cancer features are highly correlated
  (radius, perimeter, and area describe similar geometry). A coefficient's sign can
  flip when another correlated feature is added or removed. Do not over-interpret
  individual coefficients in a model full of near-duplicates.
- **Standardized coefficients:** they are expressed per standard deviation, not per
  original unit. Convert them if you need to explain the raw-unit effect.
- **Probability vs odds:** a coefficient changes the log-odds by a fixed amount,
  but the probability change depends on the starting point.
- **Not causal:** a positive coefficient means the feature is associated with the
  label in the training data. It does not mean changing the feature will change the
  outcome.

## 10. Evaluating a Classifier

### Accuracy and its limits

**Accuracy** is the fraction of rows classified correctly:

```text
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

It is easy to understand and useful when the classes are balanced and all mistakes
cost the same. It is a poor sole metric when one class dominates, because a model
that ignores the minority class can still look accurate.

### The confusion matrix

The **confusion matrix** counts every combination of true and predicted label.
Choose the positive class first; here positive means malignant.

|  | Predicted benign (0) | Predicted malignant (1) |
|--|---------------------:|------------------------:|
| **Actual benign (0)** | TN | FP |
| **Actual malignant (1)** | FN | TP |

- **TP (true positive):** actual positive, predicted positive.
- **TN (true negative):** actual negative, predicted negative.
- **FP (false positive):** actual negative, predicted positive. A false alarm.
- **FN (false negative):** actual positive, predicted negative. A missed case.

Everything else is computed from these four counts.

### Precision, recall, specificity, and F1

```text
precision   = TP / (TP + FP)     # of the ones we flagged, how many were real?
recall      = TP / (TP + FN)     # of the real positives, how many did we find?
specificity = TN / (TN + FP)     # of the real negatives, how many did we clear?
F1          = 2 * precision * recall / (precision + recall)
```

| Metric | Question it answers | High when |
|--------|---------------------|-----------|
| Accuracy | How often is the model right overall? | classes are balanced and errors cost the same |
| Precision | When the model says positive, how often is it right? | false positives are expensive |
| Recall (sensitivity) | How many real positives does the model find? | false negatives are expensive |
| Specificity | How many real negatives does the model clear? | false positives are expensive |
| F1 | A balance of precision and recall | you need one number and both matter |

There is no single best metric. The right choice depends on the problem and on the
relative cost of the two kinds of mistakes.

### Example from the practical

On 114 held-out rows (72 benign, 42 malignant), the model at threshold 0.5 produces:

```text
[[71  1]
 [ 3 39]]
```

Reading the matrix:

- 71 benign rows correctly cleared.
- 1 benign row falsely flagged as malignant.
- 3 malignant rows missed.
- 39 malignant rows correctly found.

So:

```text
accuracy    = (71 + 39) / 114 = 0.965
precision   = 39 / (39 + 1)   = 0.975
recall      = 39 / (39 + 3)   = 0.929
specificity = 71 / (71 + 1)   = 0.986
F1          = 0.951
```

That is a strong model, but the three missed cancers are the number a clinician
cares about most. Section 6's threshold discussion is exactly how we would trade
one more false alarm for fewer missed cases.

### ROC AUC and precision-recall curves

A **ROC curve** plots the true-positive rate against the false-positive rate as the
threshold moves. The **area under the curve (AUC)** summarizes how well the model
ranks positives above negatives across all thresholds. An AUC of 0.5 is random;
1.0 is perfect. On the practical split the model's AUC is about **0.996**.

AUC is threshold-independent, which is useful for comparing models, but it can look
optimistic when the positive class is rare. For imbalanced problems, a
**precision-recall curve** is often more informative. Day 18 and Day 19 cover these
tools in detail.

### Always compare with a baseline

For classification, the simplest baselines are:

- **most frequent class:** always predict the majority label;
- **prior probability:** predict the training class frequency for every row.

On the practical split, the most-frequent baseline gets 63.2% accuracy but has
**recall 0** for malignant: it never predicts the class we care about. That is the
accuracy trap in miniature.

## 11. Class Imbalance

**Class imbalance** means one class has many more rows than the other. Examples
include fraud (very rare), rare diseases, and machine failures. A 99%-negative
dataset can make a model that always says "negative" look 99% accurate while being
completely useless.

### What to do

1. **Do not use accuracy alone.** Report precision, recall, F1, and a confusion
   matrix. For rare positives, also consider precision-recall curves.
2. **Split with stratification.** `train_test_split(..., stratify=y)` keeps the
   class proportions in both parts, so the test set is representative.
3. **Use class weights.** `LogisticRegression(class_weight="balanced")` gives more
   weight to the minority class during training. It changes the model's effective
   tradeoff, not the data.
4. **Change the threshold.** Lowering it increases recall at the cost of precision.
   This is often the simplest and most transparent fix.
5. **Resample carefully.** Oversampling the minority or undersampling the majority
   can help, but only inside a training pipeline. Never resample before the split,
   and never touch the test set.
6. **Choose metrics from costs.** A missed fraud and a false fraud alert rarely
   cost the same amount. Encode that asymmetry in the threshold and the metric.

The breast cancer data is only mildly imbalanced, which is why accuracy is still
informative there. On a 1-in-1,000 fraud dataset it would not be.

## 12. Multiclass Classification

Logistic regression is binary, but it extends to `K` classes in two common ways.

### One-vs-rest (OvR)

Train one binary classifier per class: "class 1 versus everything else", "class 2
versus everything else", and so on. To predict, compute all `K` scores and choose
the class with the highest score or probability.

- Simple and parallel.
- Each classifier sees an imbalanced problem if the classes are uneven.
- Works with any binary classifier, not just logistic regression.
- The per-class probabilities do not automatically sum to 1; they are separate
  confidence scores.

### Multinomial (softmax)

Train one model with `K` weight vectors. The **softmax** function turns the `K`
scores into probabilities that sum to 1:

```text
P(class k | x) = exp(z_k) / sum_j exp(z_j)
```

- The classes compete directly with each other.
- Usually a better choice when the classes are mutually exclusive (a flower has one
  species).
- The output is a proper probability distribution over the classes.
- Training minimizes the multiclass version of log loss, called
  **categorical cross-entropy**.

| Method | How many classifiers | Output | Good when |
|--------|----------------------|--------|-----------|
| One-vs-rest | `K` binary classifiers | `K` independent scores | simplicity, reuse of binary models |
| Multinomial | one model with `K` score vectors | `K` probabilities summing to 1 | classes are mutually exclusive |

scikit-learn uses multinomial logistic regression by default for more than two
classes when the solver supports it. On the Iris data used in the practical, the
multinomial model reaches about **93.3%** test accuracy, while one-vs-rest reaches
about **90.0%**. The difference is not a law; it simply shows that the two
strategies are genuinely different models.

### Multiclass metrics

Most binary metrics extend to multiclass by averaging per-class scores:

- **Macro average:** compute the metric for each class, then take the unweighted
  mean. Every class counts equally, so rare classes matter.
- **Weighted average:** weight each class by its number of rows. Common classes
  dominate.
- **Micro average:** pool all TP, FP, and FN across classes, then compute the
  metric. Often close to accuracy for single-label multiclass problems.

Choose macro when you care about every class, and weighted when you care about
overall performance on the actual class distribution.

## 13. Assumptions and Limitations

Logistic regression is a strong, interpretable baseline, but it is not magic.

- **It is a linear classifier.** The boundary is linear in the input features. If
  the classes are separated by a curve, the model will struggle unless you add
  polynomial or interaction features, exactly as in Day 06.
- **It models the log-odds, not the probability, linearly.** This is a modeling
  choice. It is often reasonable, but it can be wrong.
- **It needs enough examples per class.** Rare classes with very few rows lead to
  unstable estimates and poor probabilities.
- **Complete separation causes trouble.** If a feature perfectly separates the
  classes, the maximum-likelihood coefficients can grow without bound. The L2
  penalty in scikit-learn keeps this finite.
- **Correlated features destabilize coefficients.** Predictions may be fine, but
  individual coefficients and odds ratios can swing wildly. Regularization helps.
- **It is sensitive to feature scaling when regularized, and to outliers.** A
  single extreme value can pull the boundary. Inspect the data.
- **Probabilities may be miscalibrated.** A model can rank rows well and still
  output probabilities that are systematically too high or too low. Calibration
  curves and methods such as Platt scaling or isotonic regression are covered
  later.
- **It cannot capture complex interactions automatically.** Trees, random forests,
  and neural networks can. Use logistic regression as an interpretable baseline,
  then test whether a more flexible model earns its complexity on held-out data.

There is no universally best classifier. The right model depends on the data, the
question, and the cost of mistakes.

## 14. Real-World Use Case: Disease Screening

A hospital wants a first-pass screening tool that flags patients who may need
follow-up testing.

- **Row:** one patient visit.
- **Features:** age, family history, lab measurements, imaging summaries.
- **Target:** `1` if the patient is later confirmed to have the disease, `0`
  otherwise.
- **Metrics:** **recall** is the headline number, because a false negative can delay
  treatment. **Precision** matters too, because false alarms create anxiety and
  cost. A confusion matrix keeps both visible.
- **Threshold:** lower than 0.5, deliberately trading some false positives for
  fewer missed cases. The exact value is chosen with clinical and operational
  stakeholders, on validation data, not on the final test set.

Why the ideas from today matter:

- **Probability, not just a label:** a triage nurse can see "82% risk" and prioritize
  accordingly, rather than a bare "positive".
- **Log loss:** it rewards honest uncertainty. A model that is 51% sure and wrong is
  treated differently from one that is 99% sure and wrong.
- **Scaling and regularization:** lab measurements have different units and are
  often correlated; standardization and a sensible `C` stabilize the fit.
- **Class imbalance:** confirmed disease may be rare, so accuracy alone would
  mislead; recall, precision, and class weights are essential.
- **Interpretability:** odds ratios help clinicians understand which measurements
  push risk up or down, subject to the cautions in Section 9.

What could go wrong:

- **Leakage:** including a lab test that is only ordered after diagnosis, or
  fitting the scaler on the full dataset before splitting.
- **Distribution shift:** a new lab machine changes the measurement scale; monitor
  and recalibrate.
- **Overconfidence:** probabilities from a small, biased sample may not transfer to
  a different hospital population.
- **Threshold drift:** a threshold chosen for one clinic's capacity may be wrong for
  another.
- **Causation:** a coefficient is not evidence that changing the feature changes
  the outcome.
- **Fairness and regulation:** a screening model used on people must be audited for
  bias and used with human oversight.

The workflow is the same as every day so far: define the question, split before any
preprocessing, establish a baseline, choose metrics from the real costs, evaluate on
unseen rows, and monitor after deployment.

## 15. Common Mistakes

1. Reporting accuracy on an imbalanced dataset and calling the model good.
2. Forgetting that the positive class must be defined explicitly before reading
   precision and recall.
3. Using the default 0.5 threshold without asking what false positives and false
   negatives cost.
4. Choosing a threshold by repeatedly scoring the test set; use a validation split.
5. Fitting a scaler on the full dataset before splitting (data leakage).
6. Forgetting to scale features before regularized logistic regression.
7. Interpreting coefficients as percentage-point changes in probability.
8. Treating a coefficient as causal evidence.
9. Over-interpreting coefficients when features are highly correlated.
10. Using a linear model on a curved boundary without adding polynomial or
    interaction features.
11. Ignoring convergence warnings and trusting an under-optimized model.
12. Comparing models trained on different splits or with different preprocessing.
13. Using squared error as the training loss and wondering why learning stalls in
    the sigmoid's flat tails.
14. Assuming one-vs-rest and multinomial produce the same probabilities.
15. Evaluating a multiclass model with a single accuracy number and no per-class
    breakdown.

## 16. Putting It Together

Before trusting a logistic regression model, ask:

1. Is the target genuinely a class, and have I defined which class is positive?
2. Did I count the classes and choose metrics that suit the balance?
3. Did I split the data before fitting any scaler or imputer?
4. Did I compare against a most-frequent or prior-probability baseline?
5. Are the features scaled, especially if I am using regularization?
6. Did the solver converge, or did I ignore a warning?
7. Does the log loss beat the prior baseline?
8. Does the confusion matrix show the mistakes I care about?
9. Is the threshold justified by the real cost of false positives and false
   negatives, and was it chosen without touching the test set?
10. Are the coefficients stable and interpretable after accounting for correlated
    features?
11. For more than two classes, did I compare one-vs-rest with multinomial and look
    at per-class metrics?
12. Am I claiming causation from an association?

If the answers are reasonable, you have a classifier you can defend.

## 17. Summary

```text
linear score -> sigmoid -> probability -> threshold -> class label
```

- Classification predicts labels; binary classification predicts one of two
  classes, usually encoded as 0 and 1.
- Logistic regression computes a linear score and squashes it with the sigmoid.
- The model is linear in the log-odds; a coefficient changes the log-odds by a
  fixed amount and multiplies the odds by `exp(coefficient)`.
- Log loss is the natural training objective for probabilities, and it punishes
  confident mistakes heavily.
- The gradient has the same shape as in linear regression:
  `(1/n) * X^T * (prediction - truth)`.
- The default decision boundary is linear and sits at probability 0.5, but the
  threshold should match the cost of mistakes.
- Accuracy is not enough when classes are imbalanced; use a confusion matrix,
  precision, recall, F1, and baseline comparisons.
- One-vs-rest and multinomial extend logistic regression to more than two classes.
- Interpretability is a strength, but coefficients are associations on the
  log-odds scale, not causal effects.

## Outcome

Learners can explain why classification needs a different model and loss from
regression, describe how the sigmoid turns a linear score into a probability,
interpret coefficients as odds ratios, explain log loss and the gradient, choose
and justify a classification threshold, read a confusion matrix with precision and
recall, handle class imbalance, and train both binary and multiclass logistic
regression with honest held-out evaluation.
