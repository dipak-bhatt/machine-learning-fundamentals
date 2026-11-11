# Day 12 — Practical: Bayes' Theorem and Naive Bayes

**Module:** Module 3: Bayesian and Instance-Based Learning

In this practical we calculate a posterior by hand, verify it with a simulation,
and then build a text classifier from counts. We will:

- Apply Bayes' theorem to a medical screening problem.
- Confirm the analytic answer with a large simulation.
- Turn a small set of labelled messages into a bag-of-words matrix.
- Train a Multinomial Naive Bayes spam classifier.
- Compare priors and posteriors.
- See Laplace smoothing change predictions for unseen words.
- Read the most informative words from the model.
- Train a Gaussian Naive Bayes model on continuous data.

Everything is bundled with scikit-learn or defined inline, so no download is
needed.

## 1. Setup

```python
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

RANDOM_STATE = 42
```

## 2. Bayes by Hand: The Medical Test

A disease affects 1% of people. The test catches 95% of sick people (sensitivity)
and correctly clears 90% of healthy people (specificity). What is `P(disease |
positive)`?

```python
prior = 0.01
sensitivity = 0.95
specificity = 0.90

p_positive_given_disease = sensitivity
p_positive_given_healthy = 1 - specificity

p_positive = (
    p_positive_given_disease * prior
    + p_positive_given_healthy * (1 - prior)
)

posterior = p_positive_given_disease * prior / p_positive

print("P(positive):", round(p_positive, 4))
print("P(disease | positive):", round(posterior, 4))
```

Expected output:

```text
P(positive): 0.1085
P(disease | positive): 0.0876
```

Only 8.8%. Most positive tests come from the large healthy group producing false
positives, not from the tiny sick group producing true positives.

The terms have names worth remembering:

| Quantity | Value |
|----------|-------|
| Prior `P(disease)` | 0.0100 |
| Likelihood `P(positive | disease)` | 0.9500 |
| Evidence `P(positive)` | 0.1085 |
| Posterior `P(disease | positive)` | 0.0876 |

Change the prior and watch the posterior move:

```python
for prevalence in [0.001, 0.01, 0.05, 0.20]:
    p_pos = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
    post = sensitivity * prevalence / p_pos
    print(f"prevalence={prevalence:.3f} -> posterior={post:.4f}")
```

Expected output:

```text
prevalence=0.001 -> posterior=0.0094
prevalence=0.010 -> posterior=0.0876
prevalence=0.050 -> posterior=0.3333
prevalence=0.200 -> posterior=0.7037
```

The same test means very different things in different populations. This is why
screening programmes target high-risk groups.

## 3. Verify with a Simulation

Simulate 200,000 people and count how often a positive test is a true positive.

```python
rng = np.random.default_rng(RANDOM_STATE)
n_people = 200_000

has_disease = rng.random(n_people) < 0.01
tests_positive = np.where(
    has_disease,
    rng.random(n_people) < 0.95,
    rng.random(n_people) < 0.10,
)

print("people with disease:", has_disease.sum())
print("positive tests:", tests_positive.sum())
print("true positives:", (has_disease & tests_positive).sum())
print("P(disease | positive):", round(has_disease[tests_positive].mean(), 4))
```

Expected output:

```text
people with disease: 2065
positive tests: 21860
true positives: 1965
P(disease | positive): 0.0899
```

The simulation gives about 9%, consistent with the analytic 8.8%. The small gap is
sampling noise. This is a good habit: when a probability result feels
counter-intuitive, simulate and check.

## 4. A Small Text Corpus

Now turn Bayes into a classifier. Define a tiny set of messages labelled `spam` or
`ham`.

```python
messages = [
    ("meeting at noon in the office", "ham"),
    ("lunch with the team today", "ham"),
    ("project report is ready for review", "ham"),
    ("call me when you get home", "ham"),
    ("see you at the meeting tomorrow", "ham"),
    ("the invoice is attached to this email", "ham"),
    ("please review the document before friday", "ham"),
    ("team lunch on thursday", "ham"),
    ("win a free iphone now", "spam"),
    ("congratulations you won a cash prize", "spam"),
    ("free money click the link now", "spam"),
    ("limited offer buy now and save", "spam"),
    ("claim your free gift today", "spam"),
    ("you have won a lottery prize", "spam"),
    ("click here to claim your cash", "spam"),
    ("urgent offer free entry now", "spam"),
]

texts = [text for text, label in messages]
labels = [label for text, label in messages]

print("messages:", len(texts))
print("spam:", labels.count("spam"), "ham:", labels.count("ham"))
```

Expected output:

```text
messages: 16
spam: 8 ham: 8
```

## 5. Bag of Words

`CountVectorizer` builds a vocabulary and counts word occurrences. It lowercases and
tokenizes by default.

```python
vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(texts)

print("document-term matrix:", X_counts.shape)
print("vocabulary size:", len(vectorizer.vocabulary_))
print("first words:", vectorizer.get_feature_names_out()[:10])
```

Expected output:

```text
document-term matrix: (16, 59)
vocabulary size: 59
first words: ['about' 'and' 'attached' 'before' 'buy' 'call' 'cash' 'claim' 'click'
 'congratulations']
```

Each row is a message; each column is a word; each cell is a count. Word order is
discarded. That loss of information is exactly what makes the multinomial model
tractable.

In a real pipeline the vectorizer must be fit on training text only. Fitting on all
text leaks the vocabulary, and with TF-IDF it also leaks the document frequencies.

## 6. Train Multinomial Naive Bayes

```python
model = MultinomialNB()
model.fit(X_counts, labels)

print("classes:", model.classes_)
print("class counts:", model.class_count_)
print("feature log-probability shape:", model.feature_log_prob_.shape)
```

Expected output:

```text
classes: ['ham' 'spam']
class counts: [8. 8.]
feature log-probability shape: (2, 59)
```

The model stores one log-probability per word per class, plus the class log-priors.
That is the entire trained model.

## 7. Priors and Posteriors

With eight messages of each class, the priors are 50/50.

```python
priors = np.exp(model.class_log_prior_)
print("prior for ham:", round(priors[0], 4))
print("prior for spam:", round(priors[1], 4))
```

Expected output:

```text
prior for ham: 0.5
prior for spam: 0.5
```

On real email the spam prior is usually much lower, and that lower prior makes the
classifier demand stronger evidence before calling a message spam.

## 8. Predict New Messages

```python
new_messages = [
    "free cash prize click now",
    "meeting tomorrow to review the report",
    "win a free gift",
    "call me tomorrow about the meeting",
]

X_new = vectorizer.transform(new_messages)

for text, prediction, proba in zip(
    new_messages, model.predict(X_new), model.predict_proba(X_new)
):
    print(f"{text!r}")
    print(f"  predicted={prediction}  P(ham)={proba[0]:.4f}  P(spam)={proba[1]:.4f}")
```

Expected output:

```text
'free cash prize click now'
  predicted=spam  P(ham)=0.0012  P(spam)=0.9988
'meeting tomorrow to review the report'
  predicted=ham  P(ham)=0.9884  P(spam)=0.0116
'win a free gift'
  predicted=spam  P(ham)=0.0426  P(spam)=0.9574
'call me tomorrow about the meeting'
  predicted=ham  P(ham)=0.9834  P(spam)=0.0166
```

The first message is dominated by spam words and is classified with near certainty.
The second is full of ham words. The third is a mixed case: "free" and "gift" push
toward spam, while "win" also appears in spam training messages, but the posterior
is less extreme. This is the ranking behaviour described in the theory.

## 9. Laplace Smoothing

The vectorizer did not see the word "bonus" during training:

```python
print("'bonus' in vocabulary:", "bonus" in vectorizer.vocabulary_)
```

Expected output:

```text
'bonus' in vocabulary: False
```

Smoothing controls how the model handles words it has never seen. Compare different
`alpha` values on a message containing "bonus".

```python
test_message = ["lottery bonus offer"]

for alpha in [0.01, 0.1, 0.5, 1.0]:
    smoothed = MultinomialNB(alpha=alpha)
    smoothed.fit(X_counts, labels)
    proba = smoothed.predict_proba(vectorizer.transform(test_message))[0]
    print(f"alpha={alpha}: P(ham)={proba[0]:.4f} P(spam)={proba[1]:.4f}")
```

Expected output:

```text
alpha=0.01: P(ham)=0.0000 P(spam)=1.0000
alpha=0.1: P(ham)=0.0037 P(spam)=0.9963
alpha=0.5: P(ham)=0.0564 P(spam)=0.9436
alpha=1.0: P(ham)=0.1336 P(spam)=0.8664
```

"bonus" is unseen, so its contribution depends on smoothing. With `alpha=0.01` the
model is nearly certain; with `alpha=1.0` it hedges more. Setting `alpha=0` would
make the unseen word's probability exactly zero and produce a warning, which is why
we never do it.

Larger smoothing is not automatically better. It can wash out real signal when the
corpus is large. Tune `alpha` with cross-validation on the training set.

## 10. Which Words Drive the Decision?

The difference between the spam and ham log-probabilities ranks words by evidence.

```python
feature_names = np.array(vectorizer.get_feature_names_out())
log_diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]

order = np.argsort(log_diff)

print("most spam-indicating words:")
print(feature_names[order[-10:]][::-1])
print("\nmost ham-indicating words:")
print(feature_names[order[:10]])
```

Expected output:

```text
most spam-indicating words:
['now' 'free' 'won' 'prize' 'offer' 'claim' 'your' 'cash' 'click' 'urgent']

most ham-indicating words:
['at' 'is' 'meeting' 'lunch' 'review' 'team' 'the' 'for' 'home' 'in']
```

These are exactly the words a human would pick. Naive Bayes is interpretable in a
way that a neural network is not: its evidence is a list of word weights.

## 11. Gaussian Naive Bayes on Continuous Data

For continuous features, use a Gaussian likelihood. The breast cancer dataset has
30 numeric measurements.

```python
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)

print("train accuracy:", round(gaussian.score(X_train, y_train), 4))
print("test accuracy:", round(gaussian.score(X_test, y_test), 4))
```

Expected output:

```text
train accuracy: 0.9460
test accuracy: 0.9371
```

Compare with logistic regression, which is a strong linear model here:

```python
logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
logistic.fit(X_train, y_train)

print("logistic test accuracy:", round(logistic.score(X_test, y_test), 4))
```

Expected output:

```text
logistic test accuracy: 0.9860
```

The independence assumption costs about five points. The features are correlated,
so Gaussian Naive Bayes overcounts evidence. It is still a fast, respectable
baseline, and it would be much cheaper on a huge dataset.

Inspect where it fails:

```python
y_pred = gaussian.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=cancer.target_names))
```

Expected output:

```text
[[46  7]
 [ 2 88]]
              precision    recall  f1-score   support

   malignant       0.96      0.87      0.91        53
      benign       0.93      0.98      0.95        90

    accuracy                           0.94       143
   macro avg       0.94      0.92      0.93       143
weighted avg       0.94      0.94       0.94       143
```

Most errors are malignant cases predicted as benign. In a screening context that is
the dangerous direction, so recall on the malignant class matters more than raw
accuracy.

## 12. Try It Yourself

1. **Change the prior.** Rebalance the messages to 4 spam and 12 ham, retrain, and
   see how the predictions and probabilities change.
2. **Use Bernoulli Naive Bayes.** Binarize the counts with
   `CountVectorizer(binary=True)` and compare accuracy and probabilities.
3. **Use TF-IDF.** Replace counts with `TfidfVectorizer` and discuss why the model
   still works.
4. **Break independence.** Add a message repeating "free" ten times and observe the
   overconfidence. Explain why the independence assumption causes it.
5. **Tune smoothing.** Run cross-validation over `alpha` values on a larger text
   corpus and report the best.
6. **Calibrate the probabilities.** Compare `predict_proba` with `CalibratedClassifierCV`
   and explain the difference.

## 13. What We Learned

- Bayes' theorem turns a prior and a likelihood into a posterior.
- The prior can dominate: a 95% sensitive test for a 1% disease gives only an 8.8%
  posterior.
- A simulation is a good check on a surprising probability.
- Naive Bayes assumes feature independence given the class, making estimation cheap.
- Multinomial Naive Bayes classifies text well despite the false assumption.
- Laplace smoothing prevents unseen words from vetoing a class.
- Spam-indicating and ham-indicating words can be read directly from the model.
- Gaussian Naive Bayes works on continuous data but loses accuracy when features are
  correlated.

## Outcome

Learners can apply Bayes' theorem and interpret prior, likelihood, evidence, and
posterior, verify a probability by simulation, build a bag-of-words text
classifier, explain the conditional-independence assumption, apply and tune Laplace
smoothing, read word-level evidence from the model, and evaluate Naive Bayes fairly
against a stronger baseline.
