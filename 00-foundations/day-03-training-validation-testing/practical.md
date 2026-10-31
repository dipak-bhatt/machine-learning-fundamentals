# Day 03 — Practical: Splitting Data and Catching Overfitting

**Module:** Module 1: Foundations

In this practical we will split a dataset into training and test sets, train
decision trees of different complexity, and see a model memorize the training
data while doing worse on new data.

We will use `students_large.csv`, a synthetic dataset of 200 students. It has
three useful features and five irrelevant features, so we can see how a complex
model can be fooled by noise.

## 1. Setup

Start Jupyter from the repository folder and create a notebook inside
`00-foundations/day-03-training-validation-testing/` (see the
[Day 01 practical](../day-01-introduction-to-ml/practical.md) if you need a
reminder).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

print("Libraries loaded")
```

## 2. Load the Dataset

```python
df = pd.read_csv("students_large.csv")
df.head()
```

If you get a `FileNotFoundError`, the notebook is not in the same folder as the
CSV file. Run Jupyter from this folder, or use the full path.

The columns are:

| Column | Meaning | Useful? |
|--------|---------|---------|
| `student_id` | A unique ID for each student | No — an ID cannot help predict passing |
| `hours_studied` | Hours studied per week | Yes |
| `attendance_percent` | Class attendance percentage | Yes |
| `previous_score` | Score on the previous exam | Yes |
| `shoe_size` | Shoe size | No |
| `coffee_cups` | Cups of coffee per day | No |
| `lucky_number` | A favorite number | No |
| `phone_battery` | Battery percentage | No |
| `bus_wait_minutes` | Minutes spent waiting for the bus | No |
| `passed` | Target: `Yes` or `No` | This is what we predict |

The five irrelevant columns are there on purpose. Real datasets often contain
columns that look like data but carry no useful signal.

Let us check the size and the target balance:

```python
print("Shape:", df.shape)
print(df["passed"].value_counts())
df.info()
```

You should see 200 rows and 10 columns, with 107 `Yes` and 93 `No` values. The
classes are reasonably balanced, so accuracy will be a fair first check.

## 3. The Cheating Experiment: Testing on Training Data

First, let us do the wrong thing on purpose. We will train a model on all the
data and then test it on the same data.

```python
X_all = df.drop(columns=["student_id", "passed"])
y_all = df["passed"]

cheating_model = DecisionTreeClassifier(random_state=42)
cheating_model.fit(X_all, y_all)

print("Training accuracy:", accuracy_score(y_all, cheating_model.predict(X_all)))
```

The answer is **1.0**. The model got every training student right.

Does that mean it is a good model? No. It had already seen every student and
every answer. This is like giving a student the practice test with the answer
key and then being impressed when they score 100%. The model may have memorized
the students instead of learning the pattern.

We need data the model has **never seen**.

## 4. Split the Data

We will set aside 40% of the rows as a test set. The model will not use them
during training.

```python
X = df.drop(columns=["student_id", "passed"])
y = df["passed"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y,
)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Training labels:", y_train.value_counts().to_dict())
print("Test labels:", y_test.value_counts().to_dict())
```

Expected output:

```
Training rows: 120
Test rows: 80
Training labels: {'Yes': 64, 'No': 56}
Test labels: {'Yes': 43, 'No': 37}
```

What the arguments mean:

- **`test_size=0.4`** keeps 40% of the rows for testing.
- **`random_state=42`** makes the split reproducible. Everyone who runs the
  same code gets the same split.
- **`stratify=y`** keeps the same proportion of `Yes` and `No` in both sets. This
  matters when one class is rare.

In a real project, we would also create a **validation set** for tuning. For
this small demonstration we will use the test set to compare a few models, but
remember that in real work the test set should be opened only once.

## 5. Train a Simple Model

A decision tree asks a series of yes/no questions. A tree with `max_depth=1` can
ask only one question, so it is very simple.

```python
simple_model = DecisionTreeClassifier(max_depth=1, random_state=42)
simple_model.fit(X_train, y_train)

train_accuracy = accuracy_score(y_train, simple_model.predict(X_train))
test_accuracy = accuracy_score(y_test, simple_model.predict(X_test))

print("Training accuracy:", round(train_accuracy, 3))
print("Test accuracy:", round(test_accuracy, 3))
```

Expected output:

```
Training accuracy: 0.858
Test accuracy: 0.863
```

The two numbers are close. This model is not perfect, but it seems to have
learned a real pattern rather than memorizing.

Let us see the question it learned:

```python
print(export_text(simple_model, feature_names=list(X_train.columns)))
```

You should see something like:

```
|--- hours_studied <= 4.80
|   |--- class: No
|--- hours_studied >  4.80
|   |--- class: Yes
```

In plain words: "If a student studied 4.8 hours or less, predict fail;
otherwise predict pass." That is a simple, understandable rule.

## 6. Make the Model More Complex

Now let us allow deeper trees. A deeper tree can ask many questions and can
memorize individual training examples.

```python
depths = [1, 2, 3, 4, 5, None]
rows = []

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    train = accuracy_score(y_train, model.predict(X_train))
    test = accuracy_score(y_test, model.predict(X_test))

    rows.append(
        {
            "max_depth": "None" if depth is None else depth,
            "train_accuracy": round(train, 3),
            "test_accuracy": round(test, 3),
            "gap": round(train - test, 3),
        }
    )

results = pd.DataFrame(rows)
results
```

Expected result (values may differ slightly by scikit-learn version):

| max_depth | train_accuracy | test_accuracy | gap |
|-----------|---------------:|--------------:|----:|
| 1 | 0.858 | 0.863 | -0.004 |
| 2 | 0.875 | 0.850 | 0.025 |
| 3 | 0.908 | 0.850 | 0.058 |
| 4 | 0.958 | 0.825 | 0.133 |
| 5 | 0.975 | 0.812 | 0.162 |
| None | 1.000 | 0.787 | 0.213 |

Notice the pattern:

- Training accuracy goes **up** as the tree gets deeper.
- Test accuracy eventually goes **down**.
- The gap between them gets **bigger**.

Depth 1 has a tiny negative gap. That can happen by chance, especially with a
small test set, so do not read too much into it.

This is **overfitting**. The unlimited tree can memorize the training data,
including the noise and the irrelevant features, but that memorization does not
help on new students.

Let us plot it:

```python
plt.figure(figsize=(7, 4))
plt.plot(results["max_depth"].astype(str), results["train_accuracy"], marker="o", label="Training")
plt.plot(results["max_depth"].astype(str), results["test_accuracy"], marker="o", label="Test")
plt.xlabel("max_depth (None = unlimited)")
plt.ylabel("Accuracy")
plt.title("Training vs Test Accuracy")
plt.legend()
plt.show()
```

The two lines separate as the model becomes more complex. The best test score
here is at a low depth, not at the deepest tree.

## 7. Underfitting: A Model That Is Too Simple

A model can also be too simple. The simplest possible classifier always predicts
the most common class and ignores all features.

```python
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)

print("Dummy train accuracy:", round(accuracy_score(y_train, dummy.predict(X_train)), 3))
print("Dummy test accuracy:", round(accuracy_score(y_test, dummy.predict(X_test)), 3))
```

Expected output:

```
Dummy train accuracy: 0.533
Dummy test accuracy: 0.537
```

Both scores are low and close together. The model is not overfitting; it is
**underfitting**. It never learned anything useful.

Another example: train a model using only `shoe_size`, which has nothing to do
with passing.

```python
shoe_model = DecisionTreeClassifier(max_depth=1, random_state=42)
shoe_model.fit(X_train[["shoe_size"]], y_train)

print("Shoe-size train accuracy:", round(accuracy_score(y_train, shoe_model.predict(X_train[["shoe_size"]])), 3))
print("Shoe-size test accuracy:", round(accuracy_score(y_test, shoe_model.predict(X_test[["shoe_size"]])), 3))
```

Expected output:

```
Shoe-size train accuracy: 0.592
Shoe-size test accuracy: 0.55
```

The scores are low because the feature does not contain the signal. A more
complex model on the same feature would still fail; the problem is the
information, not just the model.

## 8. Memorization with an ID Feature

Here is a dramatic example of overfitting. We will use only `student_id` as the
feature.

```python
X_id = df[["student_id"]]
y_id = df["passed"]

X_id_train, X_id_test, y_id_train, y_id_test = train_test_split(
    X_id, y_id, test_size=0.4, random_state=42, stratify=y_id
)

id_model = DecisionTreeClassifier(random_state=42)
id_model.fit(X_id_train, y_id_train)

print("ID-only train accuracy:", round(accuracy_score(y_id_train, id_model.predict(X_id_train)), 3))
print("ID-only test accuracy:", round(accuracy_score(y_id_test, id_model.predict(X_id_test)), 3))
```

Expected output:

```
ID-only train accuracy: 1.0
ID-only test accuracy: 0.45
```

The tree got **100% on training** and about **45% on test**. It did not learn
anything about studying; it simply memorized which IDs passed. On new students,
the IDs are new, so the model is useless.

This is why IDs, row numbers, and timestamps should almost never be features.
They let the model cheat.

## 9. Learning Curves

A learning curve shows how training and validation scores change as we give the
model more training examples. We use 5-fold cross-validation to make the curve
smoother.

> Note: for a smoother teaching example, the curves below use the full dataset.
> In a real project, draw learning curves on the training set only, so the test
> set stays untouched.

```python
def learning_curve_table(model, X, y, title):
    sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="accuracy",
        random_state=42,
        shuffle=True,
    )

    table = pd.DataFrame(
        {
            "training_examples": sizes,
            "train_accuracy": train_scores.mean(axis=1),
            "validation_accuracy": val_scores.mean(axis=1),
        }
    )

    print(title)
    print(table.round(3).to_string(index=False))

    plt.figure(figsize=(7, 4))
    plt.plot(sizes, train_scores.mean(axis=1), marker="o", label="Training")
    plt.plot(sizes, val_scores.mean(axis=1), marker="o", label="Validation")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.show()

    return table
```

### A tree with limited depth

```python
curve_depth_3 = learning_curve_table(
    DecisionTreeClassifier(max_depth=3, random_state=42),
    X,
    y,
    "Learning Curve: max_depth=3",
)
```

Expected values:

| training_examples | train_accuracy | validation_accuracy |
|------------------:|---------------:|--------------------:|
| 16 | 0.988 | 0.745 |
| 36 | 0.972 | 0.785 |
| 57 | 0.954 | 0.785 |
| 77 | 0.940 | 0.815 |
| 98 | 0.937 | 0.850 |
| 118 | 0.924 | 0.820 |
| 139 | 0.919 | 0.805 |
| 160 | 0.918 | 0.820 |

The training score slowly falls because it is harder to fit more data perfectly.
The validation score rises and then roughly levels off. The gap is small, which
is a good sign.

### An unlimited tree

```python
curve_full = learning_curve_table(
    DecisionTreeClassifier(random_state=42),
    X,
    y,
    "Learning Curve: Unlimited Tree",
)
```

Expected values:

| training_examples | train_accuracy | validation_accuracy |
|------------------:|---------------:|--------------------:|
| 16 | 1.000 | 0.755 |
| 36 | 1.000 | 0.795 |
| 57 | 1.000 | 0.825 |
| 77 | 1.000 | 0.830 |
| 98 | 1.000 | 0.845 |
| 118 | 1.000 | 0.795 |
| 139 | 1.000 | 0.790 |
| 160 | 1.000 | 0.780 |

The training score stays at **1.000** because the tree can always memorize. The
validation score is lower, and it even drops after a peak. The large gap is a
clear sign of overfitting.

### How to read the two curves

- **Limited tree:** training and validation scores move closer together as data
  grows. More data would probably help a little.
- **Unlimited tree:** training stays perfect while validation is stuck below it.
  The model is memorizing. A simpler model, better features, or more data would
  help more than just adding depth.

## 10. Data Leakage: A Warning

Imagine we fill missing values using the median of the **whole** dataset before
splitting. The test set then influences the training data, because its values
helped calculate that median. This is **data leakage**.

The safe order is:

```
1. Split the data.
2. Compute the median (or mean, scale, encoding) on the training set only.
3. Apply the same value to the validation and test sets.
```

The same idea applies to scaling, feature selection, and encoding. We will see
this again on Day 4 when we discuss the full ML workflow.

## 11. Try It Yourself

Use the `X` and `y` variables from this notebook.

1. Split with `test_size=0.2` and `random_state=0`. How do the scores change?
2. Try `max_depth=2`, `4`, and `6`. Which depth gives the best test accuracy?
3. Train a model using only `attendance_percent`. Is it underfitting or
   overfitting?
4. Train a model using only `student_id` (as in Section 8). Why is the training
   accuracy 1.0 but the test accuracy so low?
5. Plot a learning curve for `max_depth=1`. What does it tell you?
6. Split without `stratify=y` and compare the class proportions in the training
   and test sets.

Hints:

```python
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

attendance_model = DecisionTreeClassifier(max_depth=1, random_state=42)
attendance_model.fit(X_train[["attendance_percent"]], y_train)
print(accuracy_score(y_test, attendance_model.predict(X_test[["attendance_percent"]])))

id_tree = DecisionTreeClassifier(random_state=42)
id_tree.fit(X_id_train, y_id_train)
print("ID-only train:", accuracy_score(y_id_train, id_tree.predict(X_id_train)))
print("ID-only test:", accuracy_score(y_id_test, id_tree.predict(X_id_test)))

X_train3, X_test3, y_train3, y_test3 = train_test_split(
    X, y, test_size=0.2, random_state=0
)
print(y_train3.value_counts(normalize=True))
print(y_test3.value_counts(normalize=True))
```

## 12. What We Learned

In this practical we:

- Saw why testing on training data gives a misleading score.
- Split data into training and test sets with `train_test_split`.
- Compared training and test accuracy.
- Used tree depth to create underfitting and overfitting.
- Memorized student IDs to show how a model can cheat.
- Plotted learning curves and read the gap between training and validation.
- Learned why preprocessing must be fitted on the training set only.

## Outcome

Learners can split a dataset correctly, compare training and test performance,
recognize overfitting and underfitting, and use a learning curve to decide what
to try next.
