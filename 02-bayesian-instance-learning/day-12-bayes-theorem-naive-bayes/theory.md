# Day 12 — Theory: Probability, Bayes' Rule, and Naive Bayes

**Module:** Module 3: Bayesian and Instance-Based Learning

Every model so far has been a function fitted to data. A **Bayesian** model starts
from a different place: it represents beliefs as probabilities, updates those
beliefs when evidence arrives, and makes decisions from the updated beliefs. The
engine of that update is **Bayes' theorem**, one of the most useful formulas in
statistics.

This day builds the probability vocabulary needed to read the formula, applies it
to a diagnostic test where the answer is famously counter-intuitive, and then
turns it into a practical classifier. **Naive Bayes** makes one bold simplifying
assumption, yet works remarkably well for spam filtering and other text problems.
Understanding both why it is naive and why it works is the goal.

## In Plain English

**Think of it like this:** Bayes' theorem is updating a belief when new evidence
arrives — prior belief + evidence → updated belief.

- `P(A|B) = P(B|A) * P(A) / P(B)`.
- The **prior matters**: a rare disease still produces many false positives.
- **Naive Bayes** pretends features are independent given the class. That is
  usually false, yet the classifier often works well.
- It is fast and strong for text; treat its probabilities as rough, but its
  ranking as useful.
- Work with **log-probabilities** so tiny numbers do not underflow.

## 1. Probability in Plain Language

A **random experiment** produces an outcome we cannot predict with certainty. The
set of all possible outcomes is the **sample space** `S`. An **event** is a subset
of that space. For a coin:

```text
S = {heads, tails}
event "not heads" = {tails}
```

A **probability** `P(A)` assigns a number between 0 and 1 to each event, with:

- `P(S) = 1` (something must happen),
- `P(A or B) = P(A) + P(B)` when `A` and `B` cannot both happen,
- `P(not A) = 1 - P(A)`.

The **complement** rule is used constantly in diagnostics: the probability of a
negative test is one minus the probability of a positive one.

## 2. Conditional Probability

A **conditional probability** asks how likely an event is *given* that another
event happened:

```text
P(A | B) = P(A and B) / P(B),   provided P(B) > 0
```

The vertical bar is read "given". `P(A | B)` restricts attention to the world
where `B` is true and asks what fraction of that world also has `A`.

Example: roll a fair die.

- `P(odd) = 3/6 = 0.5`
- `P(odd | greater than 3) = P(odd and >3) / P(>3) = (1/6) / (3/6) = 1/3`

Knowing the result exceeds 3 removes 1, 2, and 3 from consideration and leaves
4, 5, 6, of which only 5 is odd.

Conditional probability is not symmetric. `P(spam | contains "free")` and
`P(contains "free" | spam)` are different numbers. Confusing them is the source of
many real-world errors.

## 3. Joint, Marginal, and Conditional

Three related quantities appear everywhere:

- **Joint probability** `P(A and B)`: both events happen.
- **Marginal probability** `P(A)`: one event happens, summing over the other.
- **Conditional probability** `P(A | B)`: one event given the other.

The **marginalization** rule connects joint and marginal:

```text
P(A) = sum_b P(A and B=b)
```

The **product rule** connects joint and conditional:

```text
P(A and B) = P(A | B) * P(B) = P(B | A) * P(A)
```

The product rule is the algebraic seed of Bayes' theorem. Set the two right-hand
expressions equal and divide by `P(A)`.

## 4. Independence and Conditional Independence

Two events are **independent** if knowing one tells you nothing about the other:

```text
P(A and B) = P(A) * P(B)
equivalently   P(A | B) = P(A)
```

Two events are **conditionally independent given `C`** if they become independent
once `C` is known:

```text
P(A and B | C) = P(A | C) * P(B | C)
```

This second idea is the heart of both Naive Bayes and Bayesian networks. Naive
Bayes assumes every feature is conditionally independent given the class. That
assumption is usually false, but the resulting classifier is often still useful.

## 5. Bayes' Theorem

Bayes' theorem tells us how to reverse a conditional probability:

```text
P(A | B) = P(B | A) * P(A) / P(B)
```

Each term has a name:

| Term | Name | Meaning |
|------|------|---------|
| `P(A)` | prior | belief before seeing the evidence |
| `P(B | A)` | likelihood | how likely the evidence is under `A` |
| `P(B)` | evidence | overall probability of the evidence |
| `P(A | B)` | posterior | updated belief after seeing the evidence |

The evidence `P(B)` is often computed by marginalizing over all hypotheses:

```text
P(B) = sum_a P(B | a) * P(a)
```

For two hypotheses `A` and `not A`, this becomes the familiar form:

```text
P(A | B) = P(B | A) P(A) / [ P(B | A) P(A) + P(B | not A) P(not A) ]
```

### The medical test that surprises everyone

A disease affects 1% of a population. A test is 95% sensitive (it detects 95% of
sick people) and 90% specific (it correctly clears 90% of healthy people). A
random person tests positive. What is the probability they have the disease?

```text
P(disease)     = 0.01
P(positive | disease)     = 0.95
P(positive | no disease)  = 0.10   (false positive rate)

P(positive) = 0.95 * 0.01 + 0.10 * 0.99
            = 0.0095 + 0.0990
            = 0.1085

P(disease | positive) = 0.0095 / 0.1085 = 0.0876
```

Only about **8.8%**. The false positives from the huge healthy group swamp the true
positives from the tiny sick group. The prior matters enormously. This is why a
positive result on a rare condition is usually followed by a confirmatory test.

## 6. Bayesian Decision Theory

A Bayesian classifier computes a posterior probability for each class and then makes
a decision. The simplest rule is **maximum a posteriori (MAP)**:

```text
choose the class c with the largest P(c | x)
```

Since the evidence `P(x)` is the same for every class, it can be ignored when
comparing:

```text
P(c | x)  proportional to  P(x | c) * P(c)
```

The prior `P(c)` matters. If spam is 20% of email, a message must look quite
spam-like before the posterior crosses 50%.

When mistakes have different costs, MAP is not always right. **Bayesian decision
theory** chooses the action with the lowest expected cost. Rejecting a fraudulent
transaction and annoying a legitimate customer can have very different costs, and
the threshold should reflect that.

## 7. Naive Bayes

Naive Bayes applies Bayes' theorem to a feature vector `x = (x_1, ..., x_p)` and
class `c`:

```text
P(c | x)  proportional to  P(c) * P(x_1, ..., x_p | c)
```

The joint likelihood `P(x_1, ..., x_p | c)` is hard to estimate: it needs data for
every combination of features. The **naive conditional-independence assumption**
splits it:

```text
P(x_1, ..., x_p | c) = product_j P(x_j | c)
```

Now we only need one distribution per feature per class, which is far cheaper. The
classifier becomes:

```text
P(c | x)  proportional to  P(c) * product_j P(x_j | c)
```

To classify, compute this score for every class and pick the largest. In practice,
multiply log-probabilities to avoid numerical underflow:

```text
log P(c | x) = log P(c) + sum_j log P(x_j | c) + constant
```

The constant is the same across classes, so it can be dropped. The class with the
highest score wins.

### Why the assumption is "naive"

Features are rarely independent. In text, "free" and "money" co-occur far more than
independence would predict. In medical data, two symptoms of the same disease are
correlated. The assumption is false.

Yet the classifier often works. The reason is that classification only needs the
*ranking* of class scores to be correct, not the probabilities themselves. Even when
the probabilities are badly calibrated, the highest score is often the right class.
The independence assumption also acts as a regularizer: it prevents the model from
memorizing rare feature combinations.

## 8. The Three Main Variants

The likelihood `P(x_j | c)` depends on the feature type.

**Multinomial Naive Bayes** is the text standard. Each feature is a word count, and
the model assumes counts are drawn from a multinomial distribution. It uses word
frequencies and ignores word order. This is the variant used with `CountVectorizer`
or `TfidfVectorizer`.

**Bernoulli Naive Bayes** treats each feature as binary: present or absent. It
penalizes the absence of words too, which can help on short documents.

**Gaussian Naive Bayes** assumes each feature is normally distributed within each
class:

```text
P(x_j | c) = (1 / sqrt(2 pi sigma_jc^2)) * exp(-(x_j - mu_jc)^2 / (2 sigma_jc^2))
```

It estimates a mean and variance per feature per class. It works on continuous
data such as measurements, though it assumes those features are roughly bell-shaped
and independent.

## 9. The Zero-Frequency Problem and Laplace Smoothing

Suppose the word "lottery" never appeared in the ham training messages. Then:

```text
P("lottery" | ham) = 0
```

Because the likelihood is a product, that single zero sets the whole ham score to
zero, no matter how many other words support ham. One unseen word can veto a class.
This is the **zero-frequency problem**.

**Laplace smoothing** (additive smoothing) fixes it by adding a small count to every
feature:

```text
P(x_j | c) = (count(x_j, c) + alpha) / (count(c) + alpha * V)
```

where `V` is the number of distinct features and `alpha` is the smoothing strength.

- `alpha = 1` is classic Laplace smoothing.
- `alpha < 1` is often called Lidstone smoothing.
- `alpha = 0` is no smoothing and reintroduces the zero.

The denominator includes `alpha * V` so the probabilities still sum to 1. A larger
`alpha` pulls the estimates toward uniform; a smaller `alpha` trusts the data more.

## 10. Why Naive Bayes Works for Text

Text classification is the classic Naive Bayes success story:

- **High dimensionality, small samples.** Thousands of words, maybe hundreds of
  labelled messages. Naive Bayes estimates each word independently, so it needs far
  less data than a model with feature interactions.
- **Sparse features.** Most words are absent from most documents. The independence
  assumption is wrong, but the errors are spread out.
- **Strong priors.** Spam and legitimate email have very different word
  distributions, so even a rough model separates them.
- **Fast training and prediction.** Counting words and multiplying probabilities is
  cheap, which matters for large inboxes.
- **Interpretable.** The most spam-indicating and ham-indicating words can be read
  directly from the model.

Its weakness is calibration: the probabilities are often overconfident because the
independence assumption double-counts correlated evidence. Use the ranking, and
calibrate if you need trustworthy probabilities.

## 11. Strengths and Limitations

**Strengths**

- Extremely fast to train and predict.
- Works well with high-dimensional sparse data such as text.
- Needs little data to estimate its parameters.
- Handles multiclass naturally.
- Easy to implement and interpret.
- A strong baseline for any text problem.

**Limitations**

- The independence assumption is usually false.
- Probabilities are poorly calibrated and often overconfident.
- Cannot capture feature interactions unless they are engineered.
- Sensitive to how text is tokenized and vectorized.
- Continuous features require a distributional assumption (Gaussian).
- Rare classes and unseen words need smoothing.

## 12. Real-World Use Case: Spam Filtering

A spam filter is a textbook Naive Bayes application:

- **Features:** word counts or TF-IDF values from each email.
- **Classes:** spam and ham, with a prior from the inbox mix.
- **Model:** Multinomial Naive Bayes with Laplace smoothing.
- **Prediction:** the class with the higher log-posterior.
- **Evaluation:** precision matters more than recall if false positives (deleting
  real mail) are costly. Report a confusion matrix, not just accuracy.

Modern filters combine Naive Bayes with rules, reputation signals, and learned
embeddings, but the Bayesian core remains a useful, explainable first layer.

## 13. Common Mistakes

- **Confusing `P(A | B)` with `P(B | A)`.** The prosecutor's fallacy is the famous
  version.
- **Ignoring the prior.** A rare disease makes a positive test weak evidence.
- **Using raw probabilities without logs.** Many small numbers underflow to zero.
- **Setting `alpha=0`.** One unseen word vetoes a class.
- **Trusting Naive Bayes probabilities as calibrated.** They usually are not.
- **Leaking the vectorizer.** Fit the vocabulary on training text only.
- **Evaluating text models with accuracy alone.** Use precision, recall, and F1 when
  classes are imbalanced.
- **Assuming independence is required for correctness.** It is a modelling choice
  that trades accuracy for simplicity and speed.

## 14. Summary

- Conditional probability restricts attention to a known event.
- Bayes' theorem reverses a conditional and updates a prior into a posterior.
- The prior can dominate the conclusion, as the medical test example shows.
- MAP classification picks the class with the highest posterior.
- Naive Bayes assumes features are independent given the class, making estimation
  tractable.
- Multinomial, Bernoulli, and Gaussian variants handle counts, presence, and
  continuous features respectively.
- Laplace smoothing prevents unseen features from zeroing out a class.
- Naive Bayes is fast, interpretable, and strong on text, but its probabilities are
  not trustworthy.

## Outcome

Learners can compute conditional probabilities and apply Bayes' theorem, explain
prior, likelihood, evidence, and posterior, build and interpret a Naive Bayes
classifier, explain the conditional-independence assumption and why it still works,
apply Laplace smoothing, and evaluate a text classifier with appropriate metrics.
