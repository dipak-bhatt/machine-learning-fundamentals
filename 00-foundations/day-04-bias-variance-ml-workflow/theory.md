# Day 04 — Theory: Bias, Variance, and the ML Workflow

**Module:** Module 1: Foundations

Yesterday we learned that a model can memorize the training data and still fail on
new data. Today we answer two practical questions:

1. **Why** does that happen?
2. **How** do we build a model in a careful, repeatable way?

The first question leads to the **bias-variance tradeoff**. The second leads to
the **end-to-end machine learning workflow**.

Our running example is simple: predict the selling price of a house from its
area, age, and location.

## 1. Two Ways to Be Wrong

Imagine three people throwing darts at a target.

| Archers | Where the darts land | What it means |
|---------|----------------------|---------------|
| Archer A | All darts land together, but far to the left | Consistent, but wrong aim |
| Archer B | Darts are spread all over the target | Sometimes close, sometimes far |
| Archer C | Darts land in a tight group near the center | Good aim and consistent |

Machine learning models can fall into the same two traps:

- **Bias** is like Archer A: the model is consistently wrong because it is too
  simple or makes the wrong assumption.
- **Variance** is like Archer B: the model's answer changes a lot depending on
  which training examples it happened to see.
- A good model is like Archer C: close to the truth and stable.

There is also the **wind** pushing every dart a little. That is **irreducible
noise**: randomness in the real world that no model can remove.

In one sentence:

```
Total expected error = bias² + variance + irreducible noise
```

We cannot make noise zero. We try to keep bias and variance small at the same
time, but reducing one often increases the other. That tension is the
**bias-variance tradeoff**.

## 2. Bias: Too Simple, Consistently Wrong

**Bias** is error caused by a model that is too simple, or by an assumption that
does not match the real pattern. A high-bias model **underfits**: it does poorly
on both training and test data.

### A simple example

Suppose house prices grow faster as houses get bigger:

| Area (sqft) | Actual price |
|------------:|-------------:|
| 600 | 45,000 |
| 1,200 | 75,000 |
| 1,800 | 120,000 |
| 2,400 | 180,000 |
| 3,000 | 250,000 |

Now imagine a very simple model that always predicts the **average price**,
around 134,000. It is not crazy, but it misses the pattern completely. Small
houses are overpriced by the model, and large houses are badly underpriced.

Another simple model uses a straight line:

```
price = 30,000 + 60 × area
```

| Area | Actual | Straight-line prediction | Prediction − Actual |
|-----:|-------:|-------------------------:|-------------------:|
| 600 | 45,000 | 66,000 | +21,000 |
| 1,200 | 75,000 | 102,000 | +27,000 |
| 1,800 | 120,000 | 138,000 | +18,000 |
| 2,400 | 180,000 | 174,000 | -6,000 |
| 3,000 | 250,000 | 210,000 | -40,000 |

The line is **consistently wrong in a pattern**. It cannot bend upward, so it
overshoots small houses and undershoots large ones. That is **high bias**.

### Signs of high bias

- Training error is high.
- Test error is also high.
- The two errors are close together.
- Adding more data does not help much.

### Common causes

- The model is too simple (for example, a straight line for a curved pattern).
- Important features are missing.
- The model is too heavily regularized.
- The pattern cannot be learned from the available data.

### Common fixes

- Use a more flexible model.
- Add better features.
- Reduce regularization.
- Train longer, if the model is still improving.

## 3. Variance: Too Sensitive, Different Answers

**Variance** is error caused by a model that is too sensitive to the particular
training sample. A high-variance model **overfits**: it fits the training data
very closely, including noise and accidental quirks, but it fails on new data.

### A simple example

Imagine a model that memorizes every house it has seen. If a 1,700 sqft house
sold for 150,000 because of an unusual negotiation, the model may believe that
**all** 1,700 sqft houses are worth 150,000.

Now train the same complex model on two slightly different samples:

- Sample 1 predicts the house is worth **90,000**.
- Sample 2 predicts the same house is worth **140,000**.

The model changed a lot because of a few different training rows. That is **high
variance**. A simple model might give 110,000 both times, but if the true value
is 120,000, being stable and wrong is still not good enough.

### Signs of high variance

- Training error is very low, often near zero.
- Test error is much higher.
- The gap between training and test error is large.
- Small changes in the data cause large changes in predictions.

### Common causes

- The model is too complex.
- There is too little training data.
- There are many irrelevant features.
- The labels contain noise or mistakes.

### Common fixes

- Collect more data.
- Use a simpler model.
- Remove irrelevant features.
- Use regularization.
- Use early stopping.
- Use cross-validation for a more reliable estimate.

## 4. Irreducible Noise: The Error We Cannot Remove

Some randomness is simply part of the world:

- Two identical houses sell for different prices because of timing or
  negotiation.
- Two patients with the same symptoms have different outcomes.
- Two students with the same study habits get different exam scores.

No model can predict these differences perfectly. This is **irreducible noise**
(also called **Bayes error**). If a model reaches zero training error, it is
often memorizing noise rather than learning the signal.

> A perfect training score is a warning sign, not a goal.

## 5. The Bias-Variance Tradeoff

As we make a model more complex, two things usually happen:

- **Bias goes down.** The model can represent more complicated patterns.
- **Variance goes up.** The model becomes more sensitive to noise in the sample.

Training error usually falls as complexity increases. Validation error usually
forms a **U-shape**: it falls at first, reaches a minimum, then rises again.

```
Error
  |
  |  total error
  |   \        /
  |    \      /
  |     \    /
  |      \  /
  |       \/
  |     bias² decreases
  |     variance increases
  +----------------------------→ Model complexity
      underfit   good fit   overfit
```

| Model | Training error | Validation error | Main problem |
|-------|---------------:|-----------------:|--------------|
| Too simple | High | High | Bias (underfitting) |
| Good fit | Low | Low | Small gap |
| Too complex | Very low | High | Variance (overfitting) |

The best model is not the one with the lowest training error. It is the one with
the best performance on **unseen data**.

## 6. A Concrete House Price Example

Using the dataset from today's practical, we can compare models on the same
training and test split. The error is measured with **RMSE** (root mean squared
error): a lower number means predictions are closer to the truth.

| Model | Training RMSE | Test RMSE | What it shows |
|-------|--------------:|----------:|---------------|
| Always predict the average price | 56,420 | 58,015 | High bias: both errors are high |
| Very simple tree, depth 1 | 28,533 | 29,158 | Still high bias: it can only ask one question |
| Balanced tree, depth 4 | 7,319 | 11,497 | Good fit: low errors and a small gap |
| Unlimited tree | 1,380 | 12,961 | High variance: almost perfect on training, worse on test |

The unlimited tree looks amazing on training data: its error is only 1,380. But
its test error is **worse** than the balanced tree. It memorized the training
houses instead of learning the general pattern.

That is the whole lesson in one table: **we choose a model by validation or test
performance, not by training performance.**

> For this small lesson we compare on the test set. In a real project, use a
> validation set for these decisions and reserve the test set for one final
> check.

## 7. How to Control Complexity

There are two directions of repair.

### If the model has high bias (underfitting)

- Use a more flexible model.
- Add useful features.
- Reduce regularization.
- Train longer, if the model is still improving.

### If the model has high variance (overfitting)

- Collect more data.
- Use a simpler model.
- Remove irrelevant features.
- Use regularization.
- Limit model size (for example, tree depth).
- Use early stopping.

### Regularization in plain words

**Regularization** is a penalty for being too complicated. It discourages a
model from relying on tiny, unstable patterns.

- **Ridge regression** shrinks large coefficients toward zero.
- **Lasso regression** can shrink some coefficients all the way to zero.
- **Tree limits** such as `max_depth` and `min_samples_leaf` stop a tree from
  memorizing individual rows.

A little regularization reduces variance. Too much increases bias. We tune it
with a validation set.

## 8. Reproducibility and Random Seeds

Machine learning involves randomness:

- Splitting data into training and test sets.
- Shuffling rows.
- Initializing some models.
- Sampling rows for cross-validation.

If we do not control this randomness, two runs can give different results, and
we cannot tell whether a change helped or whether we simply got lucky.

A **random seed** makes the randomness repeatable:

```python
train_test_split(X, y, test_size=0.2, random_state=42)
DecisionTreeRegressor(max_depth=4, random_state=42)
```

With `random_state=42`, everyone who runs the same code gets the same split and
the same result.

Important: a seed does **not** make a bad model good. It only makes the
experiment repeatable. When comparing two models, use the **same** split and the
same seed for both.

## 9. Data Leakage: Cheating Without Knowing It

**Data leakage** happens when information from the test set, or from the future,
sneaks into training. The model then looks better than it really is.

### Example 1: Filling missing values before splitting

Wrong:

```python
leaky = df.copy()
median_age = leaky["age_years"].median()
leaky["age_years"] = leaky["age_years"].fillna(median_age)

X_leaky = leaky.drop(columns=["price_usd"])
y_leaky = leaky["price_usd"]
X_train_leaky, X_test_leaky, y_train_leaky, y_test_leaky = train_test_split(
    X_leaky, y_leaky, random_state=42
)
```

The test set helped calculate the median, so information leaked into training.

Right:

```python
X = df.drop(columns=["price_usd"])
y = df["price_usd"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)

median_age = X_train["age_years"].median()
X_train["age_years"] = X_train["age_years"].fillna(median_age)
X_test["age_years"] = X_test["age_years"].fillna(median_age)
```

Fit the preprocessing on the training set only, then apply it to validation and
test data.

### Other common leaks

- Scaling or encoding before splitting.
- The same patient, customer, or user appears in both training and test sets.
- Duplicate rows are split across sets.
- A feature contains the answer (for example, "was refunded" when predicting
  "will churn").
- Future information is used to predict the past.
- The test set is used again and again for tuning.

### The safe rule

```
Split first.
Fit preprocessing on training data only.
Use validation data for decisions.
Touch the test set once, at the very end.
```

## 10. The End-to-End ML Workflow

A model is only one part of a project. The full workflow looks like this:

```
1. Define the problem
        ↓
2. Collect data
        ↓
3. Explore and clean data
        ↓
4. Split into train / validation / test
        ↓
5. Build preprocessing
        ↓
6. Train models
        ↓
7. Evaluate and tune
        ↓
8. Final test
        ↓
9. Deploy and monitor
```

### 1. Define the problem

Decide what question the model should answer and how success will be measured.

For house prices:

- **Question:** What will this house sell for?
- **Target:** Sale price.
- **Metric:** RMSE, because large price errors are especially costly.
- **Decision:** Should we list the house, or wait?

A clear problem definition prevents building a model that is accurate but
useless.

### 2. Collect data

Gather rows and features. Ask:

- Does each row represent one real thing (one house, one customer)?
- Are the features available at prediction time?
- Is the sample representative of the real world?

### 3. Explore and clean data

Look at missing values, outliers, duplicates, data types, and distributions. This
is where we notice that a house age of 300 years might be a typo, or that one
neighborhood has only two sales.

### 4. Split the data

Set aside test data before looking at it. Use a validation set for tuning. For
time-based problems, split by time:

```
Train on older sales → validate on recent sales → test on the newest sales
```

A random split can leak future information into the past.

### 5. Build preprocessing

Handle missing values, encode categories, and scale features. Use a pipeline so
the same steps are applied consistently to training, validation, and test data.

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, ["area_sqft", "age_years"]),
        ("cat", categorical_pipeline, ["location"]),
    ]
)
```

The pipeline is fitted on the training data only.

### 6. Train models

Start simple:

1. A baseline (predict the average).
2. A simple model (for example, a shallow tree).
3. A more flexible model.

Always compare against the baseline. A fancy model that cannot beat "predict the
average" is not useful.

### 7. Evaluate and tune

Use the validation set to compare models, choose complexity, and tune
regularization. Look at both the score and the gap between training and
validation performance.

### 8. Final test

Evaluate the chosen model **once** on the test set. Report that number honestly.
Do not keep changing the model after seeing the test score.

### 9. Deploy and monitor

In the real world, data changes. Monitor the model:

- Are inputs still similar to training data?
- Is accuracy dropping?
- Has the world changed (prices, customer behavior, spam tactics)?

Retrain when needed.

## 11. Real-World Use Case: Employee Attrition

A company wants to predict which employees are likely to leave.

- **Row:** one employee.
- **Features:** tenure, salary, role, overtime hours, satisfaction score.
- **Target:** left or stayed.
- **Metric:** recall, if the goal is to help employees before they leave.

Where bias and variance appear:

- **High bias:** using only salary, which misses role, workload, and satisfaction.
- **High variance:** a deep tree memorizes a few unusual employees, including
  their employee IDs.
- **Noise:** an employee leaves for personal reasons that are not in the data.

Where leakage appears:

- Using the **exit interview date** as a feature.
- Using the **employee ID**, which lets the model memorize individuals.
- Using information collected **after** the employee left.
- Splitting the same employee's records across training and test sets.

A careful workflow:

1. Define the decision: who should receive support?
2. Collect historical employee records.
3. Remove IDs and future information.
4. Split by employee and, ideally, by time.
5. Fit preprocessing on training data only.
6. Compare a baseline, a simple model, and a complex model.
7. Choose the model with the best validation performance, not the best training
   score.
8. Test once and report the result.
9. Monitor as the company and job market change.

The same pattern applies to house prices, spam filtering, medical diagnosis, and
customer churn. The workflow is always more important than the algorithm.

## 12. Common Mistakes

1. Choosing a model because it has the lowest training error.
2. Comparing models on different splits.
3. Tuning on the test set.
4. Fitting preprocessing on the whole dataset before splitting.
5. Adding IDs, row numbers, or future information as features.
6. Treating a perfect training score as success.
7. Using a complex model when a simple one performs just as well.
8. Changing the random seed until the result looks good.
9. Forgetting that the real world changes over time.
10. Reporting a result without saying how the data was split.

## 13. Putting It Together

Before trusting a model, ask:

1. Is the training error low **and** the validation error low?
2. Is the gap between them small?
3. Did the model beat a simple baseline?
4. Is the model too simple or too complex?
5. Did I split the data before doing anything else?
6. Did I fit preprocessing on the training set only?
7. Could any feature leak the answer or the future?
8. Did I use the test set only once?
9. Can I reproduce the result with the same random seed?
10. Will the model still work when the world changes?

## Outcome

Learners can explain bias, variance, and irreducible noise, recognize
underfitting and overfitting, describe how regularization and data size affect
the tradeoff, avoid data leakage, and describe the complete machine learning
workflow from problem definition to monitoring.
