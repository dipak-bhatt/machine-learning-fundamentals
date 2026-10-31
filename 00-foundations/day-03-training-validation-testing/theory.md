# Day 03 — Theory: Training, Validation, Testing, and Generalization

**Module:** Module 1: Foundations

Yesterday we learned how to look at data. Today we ask a different question:
**did the model actually learn, or did it only memorize?**

This is one of the most important ideas in machine learning. A model that scores
100% on the data it has already seen can still be useless on new data.

## 1. The Problem: A Model Can Memorize Instead of Learn

Imagine two students preparing for a math exam.

- **Ana** memorizes the answers to 20 practice questions. She scores 20/20 on
  the practice test. On the real exam, the questions use the same ideas but
  different numbers, and she scores 8/20.
- **Ben** learns the method. He scores 17/20 on the practice test. On the real
  exam he scores 17/20.

Who is the better student? **Ben.** Ana memorized the practice questions; she did
not learn how to solve new ones.

Machine learning has the same trap:

```
Training data (seen)  →  model  →  high score
New data (unseen)     →  model  →  maybe a low score
```

The goal is not a high score on old data. The goal is a model that works on
**new, unseen data**. That ability is called **generalization**.

## 2. Three Sets: Training, Validation, and Test

To check generalization, we split the data into three parts.

| Set | School analogy | Used for | Typical size |
|-----|----------------|----------|--------------|
| **Training set** | Practice questions with the answer key | Learning patterns | 60–80% |
| **Validation set** | A mock exam | Tuning choices | 10–20% |
| **Test set** | The final exam | One honest final check | 10–20% |

```
All data
  ├── Training set   → the model learns from this
  ├── Validation set → we use this to compare and tune models
  └── Test set       → we open this once, at the very end
```

For example, with 1,000 rows, a 60/20/20 split gives 600 training rows, 200
validation rows, and 200 test rows.

Why do we need a separate validation set? Because if we keep changing the model
until it does well on the test set, the test set is no longer a fair exam. It is
like peeking at the final exam while you study: your final score stops being an
honest measure of what you learned.

Common splits are **60/20/20**, **70/15/15**, or **80/10/10**. For very small
datasets, we use **cross-validation** instead, which we will study on Day 19.

## 3. Training Error vs Test Error

- **Training error:** the error on the data the model learned from.
- **Test error:** the error on data the model has never seen.
- **Generalization gap:** how much worse the test score is than the training
  score.

For classification, **error = 1 − accuracy**. A model with 90% accuracy has a
10% error.

Example:

| Model | Training accuracy | Test accuracy | What it tells us |
|-------|------------------:|--------------:|------------------|
| A | 99% | 60% | Memorized the training data (overfitting) |
| B | 85% | 84% | Learned useful patterns (generalizes well) |
| C | 55% | 54% | Too simple to learn the pattern (underfitting) |

Model A looks amazing on paper, but Model B is the one we would trust in the real
world. **A low training error by itself proves nothing.**

## 4. Overfitting: Memorizing the Noise

**Overfitting** means the model fits the training data too closely, including its
noise, outliers, and accidental quirks. It does well on old data but poorly on
new data.

### A simple example

Suppose we want to predict whether a student passes. If we give the model each
student's **ID number** as a feature, a deep model can simply memorize:

- Student 1 → passed
- Student 2 → failed
- Student 3 → passed
- and so on.

On the training data it scores **100%**, because it has seen every ID. On new
students it scores about **45%**, because the IDs are new and the model never
learned anything real about studying.

This is why IDs, row numbers, and timestamps should almost never be features.
They let the model cheat.

### Signs of overfitting

- Training score is very high.
- Validation or test score is much lower.
- The gap grows as the model becomes more complex.
- The model performs badly on slightly different examples.

### Common causes

- The model is too complex (for example, a very deep decision tree).
- There is too little training data.
- There are many irrelevant features.
- The labels contain noise or mistakes.

### Common fixes

- Collect more data.
- Use a simpler model.
- Remove irrelevant features.
- Use regularization (Day 4 and later).
- Use early stopping for models that train in steps.
- Use cross-validation to get a more reliable estimate (Day 19).

## 5. Underfitting: Too Simple to Learn

**Underfitting** is the opposite problem. The model is too simple, or the
features do not contain enough information, so it performs poorly on both
training and test data.

### A simple example

Suppose we try to predict exam success using only **shoe size**. Shoe size has
nothing to do with passing. A simple model trained on it scores about **59%** on
training and **55%** on test. Both scores are low, and they are close together.

A student analogy: a student who only learns "when in doubt, choose B" will do
badly on both the practice test and the real exam.

### Signs of underfitting

- Training score is low.
- Test score is also low.
- The two scores are close to each other.

### Common causes

- The model is too simple.
- Important features are missing.
- The model is too heavily regularized.
- The pattern is not learnable from the available data.

### Common fixes

- Use a more flexible model.
- Add better features.
- Reduce regularization.
- Train longer, if the model improves over time.

## 6. The Sweet Spot: Generalization

The best model is neither too simple nor too complex.

```
Underfitting          Good fit             Overfitting
too simple            just right           too complex
high training error   low training error   very low training error
high test error       low test error       high test error
```

As model complexity increases, training error usually goes **down**. Validation
error usually forms a **U-shape**: it falls at first, reaches a minimum, then
rises again.

```
Error
  |
  |  training error \
  |                  \________
  |                           \______
  |        validation error
  |     ___/        \___
  | ___/                \___
  +--------------------------------→ Model complexity
       underfit    good fit    overfit
```

We want the bottom of the U. This is called the **approximation-generalization
tradeoff**. We will look at it more closely on Day 4 when we study bias and
variance.

## 7. Why Can a Model Generalize? (Feasibility of Learning)

We cannot test every future case. So why should a model work at all? We rely on
three assumptions:

1. **Future data is similar to training data.** The new examples come from the
   same kind of situation, not a completely different world.
2. **The training sample is representative.** It is not biased toward one group,
   one time, or one place.
3. **The pattern is learnable.** The target is not pure randomness.

In plain words: if the world stays similar and the sample is fair, patterns
learned on the sample tend to work on new data.

These assumptions can break:

- **Distribution shift:** shopping habits change after a pandemic, so a model
  trained on old data may fail.
- **Sampling bias:** training only on morning customers may not represent
  evening customers.
- **Concept drift:** spam emails change over time, so an old spam filter becomes
  outdated.

There is no guarantee that a model will always generalize. We monitor models and
retrain them when the world changes.

## 8. Where Does the Error Come From?

A useful way to think about error is:

```
Total error = approximation error + estimation error + irreducible noise
```

| Part | Plain meaning | Example | What helps |
|------|---------------|---------|------------|
| **Irreducible noise** | Randomness that no model can remove | Two patients with the same symptoms have different outcomes | Better measurements; accept a limit |
| **Approximation error** | The model family is too simple | Using a straight line for a curved pattern | A more flexible model or better features |
| **Estimation error** | The model fits quirks of this particular sample | A deep tree memorizing 20 students | More data, a simpler model, regularization |

Think of an archer:

- **Approximation error (bias):** aiming at the wrong target.
- **Estimation error (variance):** arrows scattered around the target.
- **Noise:** wind pushing every arrow a little.

A **generalization bound** is a mathematical promise of the form: "with enough
data and a model that is not too flexible, the test error will be close to the
training error with high probability." We do not need the formula today. The
intuition is enough: **more representative data and a suitably simple model make
the gap smaller.**

## 9. Learning Curves

A **learning curve** plots training and validation performance against the number
of training examples.

```
Score
  |
  |  training score \________
  |                          \____
  |  validation score   _____/
  |                   __/
  |
  +--------------------------------→ Training set size
```

With a small training set, the model can fit the data almost perfectly, so the
training score starts high. As the training set grows, the training score usually
falls a little, while the validation score rises and then levels off.

How to read one:

- **Both scores low and close together** → underfitting. More data alone will
  not help much; the model or features need to improve.
- **Training score high, validation score much lower, big gap** → overfitting.
  More data often helps close the gap.
- **Both scores high and close together** → good fit.

Learning curves help us decide between two actions:

1. Collect more data.
2. Improve the model or the features.

## 10. A Safe Splitting Workflow

Here is a simple, safe order of steps:

1. Clean obvious data errors.
2. Set aside the **test set** first.
3. Split the rest into **training** and **validation** sets.
4. Fit preprocessing (mean, median, scaler, encoder) on the **training set only**.
5. Train on the training set.
6. Compare and tune using the validation set.
7. When you are happy, evaluate **once** on the test set.
8. Report that test result. Do not keep tuning on it.

### Data leakage

**Data leakage** happens when information from the test set (or from the future)
sneaks into training. The model then looks better than it really is.

Example:

- **Wrong:** fill missing values using the median of the whole dataset, then
  split into train and test.
- **Right:** compute the median on the training set, then use that same median
  for the validation and test sets.

Other common leaks:

- The same patient appears in both training and test sets.
- Duplicate rows are split across sets.
- A feature contains the answer (for example, "was refunded" when predicting
  "will churn").
- Future information is used to predict the past.

## 11. ML Use Cases for Today's Concepts

| Concept | How it is used in ML | Real use case |
|---------|----------------------|---------------|
| Training, validation, test | Separate learning, tuning, and final checking | A hospital trains on past patients, tunes on a validation group, and reports results on new patients |
| Overfitting | Detect a model that memorizes instead of learning | A deep tree memorizes 50 recent house sales and fails on new houses |
| Underfitting | Detect a model that is too simple | Predicting house prices using only the number of bedrooms |
| Learning curves | Decide whether to collect more data or improve the model | A spam filter's validation score plateaus, so the team adds better features |
| Generalization | Judge whether the model will work in the real world | A model trained in one city is tested in another city |
| Data leakage | Keep the test score honest | Scaling features before splitting leaks test information |

### Use case 1: Medical diagnosis

A hospital wants a model that predicts whether a scan shows a disease.

- **Row:** one patient.
- **Target:** disease or no disease.
- **Split by patient, not by image.** If two images from the same patient appear
  in both training and test sets, the model may memorize the patient rather than
  the disease.
- **Overfitting risk:** many scan features and only a few patients.
- **Underfitting risk:** using only age and gender when the scan contains the
  real signal.
- **Noise:** two doctors may label the same scan differently.
- **Distribution shift:** a new scanner or a new hospital can change the images.
- **Test set:** keep it untouched until the final evaluation.

### Use case 2: House price prediction

A real-estate company wants to predict selling prices.

- **Features:** area, bedrooms, location, age.
- **Overfitting:** a very deep tree memorizes a small set of recent sales and
  fails on new houses.
- **Underfitting:** using only the number of bedrooms ignores location and size.
- **Time-based split:** train on older sales and test on newer sales, because the
  market changes over time. A random split can leak future information.
- **Learning curve:** if the validation score plateaus, more data may not help;
  better features may be the answer.
- **Noise:** unusual sales, such as a house sold between relatives, can mislead
  the model.

### Use case 3: Spam filter

An email provider wants to classify email as spam or not spam.

- **Overfitting:** memorizing exact sender addresses or rare words instead of
  general spam patterns.
- **Underfitting:** using only email length.
- **Validation set:** used to choose the decision threshold (for example, 0.5 or
  0.9).
- **Test set:** used once for the final estimate.
- **Concept drift:** spam changes, so the filter must be retrained regularly.
- **Data leakage:** using the spam label itself, or future user reports, as a
  feature.

## 12. Common Mistakes

1. Testing on the training data.
2. Tuning the model on the test set.
3. Randomly splitting data when rows are related (same patient, same user, same
   day).
4. Fitting preprocessing on the whole dataset before splitting.
5. Judging a model by accuracy alone when the classes are imbalanced.
6. Trusting a score from a very small test set.
7. Assuming a good test score will last forever.
8. Adding IDs, row numbers, or timestamps as features.

## 13. Putting It Together

Before trusting a model, ask:

1. Did I split the data before looking at the test set?
2. Is the split representative of the real world?
3. Is the training score much higher than the validation or test score?
4. Are both training and validation scores low?
5. What does the learning curve say?
6. Could any feature leak the answer?
7. Did I evaluate on the test set only once?
8. Will the world change in a way that makes the model outdated?

## Outcome

Learners can explain why models must be tested on unseen data, describe
overfitting and underfitting, read a learning curve, and avoid common mistakes
when splitting data.
