# Day 02 — Practical: Exploring Data with Python

**Module:** Module 1: Foundations

In this practical we will load a small CSV file, summarize it with statistics,
draw charts, find missing values and outliers, and calculate a simple
conditional probability.

The dataset `students.csv` is in this folder. It is a small teaching dataset
based on realistic student records. It contains 16 students with their study
hours, attendance, previous score, final score, city, and whether they passed.

## 1. Open a Notebook and Import the Libraries

Start Jupyter from the repository folder (see
[Day 01 practical](../day-01-introduction-to-ml/practical.md) if you need a
reminder), then create a notebook inside
`00-foundations/day-02-data-statistics-probability/`.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries loaded")
```

Notes:

- **NumPy** is for numbers and arrays.
- **pandas** is for tables.
- **Matplotlib** and **seaborn** are for charts.

## 2. Load the CSV File

```python
df = pd.read_csv("students.csv")
df.head()
```

`read_csv` reads the file into a **DataFrame** (pandas' name for a table).
`head()` shows the first five rows.

If you get a `FileNotFoundError`, the notebook is not in the same folder as the
CSV file. Either move the notebook to this folder or use the full path:

```python
df = pd.read_csv("D:/ws_github/d2dex/machine-learning-fundamentals/00-foundations/day-02-data-statistics-probability/students.csv")
```

## 3. First Look at the Data

```python
print("Shape:", df.shape)
print("Columns:", list(df.columns))
df.info()
```

What to look for:

- **Shape** `(16, 7)` means 16 rows and 7 columns.
- **Columns** are the feature names.
- **`df.info()`** shows each column's data type and how many non-missing values
  it has.

Notice that `city` and `passed` are text (`object`), while the rest are numbers.
`attendance_percent`, `previous_score`, and `final_score` show fewer than 16
non-null values, which means some cells are missing.

## 4. Missing Values

```python
df.isnull().sum()
```

This counts missing values per column. The output shows one missing value in
`attendance_percent`, one in `previous_score`, and one in `final_score`.

We can also see the exact rows:

```python
df[df.isnull().any(axis=1)]
```

`any(axis=1)` asks: "does this row have at least one missing value?"

### Filling missing values (one common option)

We will work on a copy so the original data stays untouched.

```python
clean = df.copy()

clean["attendance_percent"] = clean["attendance_percent"].fillna(
    clean["attendance_percent"].median()
)
clean["previous_score"] = clean["previous_score"].fillna(
    clean["previous_score"].median()
)

clean.isnull().sum()
```

Only `final_score` is still missing. In a real project, think carefully before
filling the **target** column, because you would be inventing the answer you
want to predict. For this demonstration we will fill it with the median too so
the charts are easier to read.

```python
clean["final_score"] = clean["final_score"].fillna(clean["final_score"].median())
clean.isnull().sum()
```

Now every column has zero missing values.

> Important: filling with the median is only one option. Sometimes it is better
> to remove the row, or to ask why the value is missing in the first place.

## 5. Descriptive Statistics

pandas can calculate many statistics in one line:

```python
clean.describe()
```

`describe()` summarizes the numeric columns with count, mean, standard
deviation (`std`), minimum, 25th percentile, median (`50%`), 75th percentile,
and maximum.

We can also calculate individual statistics:

```python
print("Mean final score:", clean["final_score"].mean())
print("Median final score:", clean["final_score"].median())
print("Mode of city:", clean["city"].mode()[0])
print("Standard deviation of final score:", clean["final_score"].std())
```

Expected values (approximately, after filling the missing final score with its
median):

- Mean final score ≈ 65.1
- Median final score = 65.0
- Mode of city = Kathmandu
- Standard deviation ≈ 17.6

### Mean vs median with an outlier

Look at `hours_studied`. One student reported studying 40 hours.

```python
print("Mean hours:", clean["hours_studied"].mean())
print("Median hours:", clean["hours_studied"].median())
```

The mean is about **7.6**, but the median is **5.5**. The single value `40`
pulls the mean upward. This is a small, clear example of why the median can be a
safer summary when outliers are present.

### Categorical summaries

For text columns, counting is more useful than averaging:

```python
print(clean["city"].value_counts())
print(clean["passed"].value_counts())
```

`value_counts()` tells us how many rows belong to each category.

## 6. Visualizing Distributions

### Histogram

A histogram groups numbers into bins and shows how many values fall in each
bin.

```python
plt.figure(figsize=(7, 4))
plt.hist(clean["final_score"], bins=8, edgecolor="black")
plt.title("Distribution of Final Scores")
plt.xlabel("Final score")
plt.ylabel("Number of students")
plt.show()
```

Questions to ask:

- Where is the center?
- Is the shape roughly symmetric, or does it have a tail?
- Are there gaps or unusual values?

### Histogram of study hours

```python
plt.figure(figsize=(7, 4))
plt.hist(clean["hours_studied"], bins=8, edgecolor="black")
plt.title("Distribution of Study Hours")
plt.xlabel("Hours studied")
plt.ylabel("Number of students")
plt.show()
```

The bar far to the right is the student who reported 40 hours. That is a clue
that we should investigate an outlier.

### Box plot

A box plot shows the median, the middle 50% of the data, and possible outliers.

```python
plt.figure(figsize=(7, 4))
sns.boxplot(x=clean["final_score"])
plt.title("Box Plot of Final Scores")
plt.xlabel("Final score")
plt.show()
```

Reading a box plot:

- The line inside the box is the **median**.
- The box goes from **Q1** to **Q3** (the middle 50%).
- The whiskers show the normal range.
- Dots beyond the whiskers are possible **outliers**.

## 7. Finding Outliers with the IQR Rule

A common rule flags values below `Q1 − 1.5 × IQR` or above `Q3 + 1.5 × IQR`.

```python
q1 = clean["hours_studied"].quantile(0.25)
q3 = clean["hours_studied"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("Lower limit:", lower)
print("Upper limit:", upper)

outliers = clean[(clean["hours_studied"] < lower) | (clean["hours_studied"] > upper)]
outliers
```

You should see the student with `hours_studied = 40` flagged as an outlier.

Before deleting it, ask: is it a typo, or a real value? In this file it is almost
certainly a data-entry mistake (40 hours of study is possible, but unlikely
alongside a low final score). For learning purposes we will keep it and simply
note that it affects the mean.

## 8. Relationships Between Variables

### Scatter plot

A scatter plot puts one feature on the x-axis and another on the y-axis. Each dot
is one row.

```python
plt.figure(figsize=(7, 4))
sns.scatterplot(data=clean, x="hours_studied", y="final_score", hue="passed")
plt.title("Study Hours vs Final Score")
plt.xlabel("Hours studied")
plt.ylabel("Final score")
plt.show()
```

Look for a pattern: do students who study more tend to score higher? The colors
show who passed and who did not. The dot far to the right is the 40-hour
outlier. One unusual value can visibly change what the chart seems to say, which
is why we inspect data before trusting a correlation.

### Correlation

```python
clean[["hours_studied", "attendance_percent", "previous_score", "final_score"]].corr()
```

The result is a **correlation matrix**. Each value is between −1 and +1:

- Close to +1 → strong positive relationship.
- Close to 0 → weak or no linear relationship.
- Close to −1 → strong negative relationship.

The diagonal is always 1 because every column is perfectly correlated with
itself.

### Correlation heatmap

```python
plt.figure(figsize=(7, 5))
sns.heatmap(
    clean[["hours_studied", "attendance_percent", "previous_score", "final_score"]].corr(),
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
)
plt.title("Correlation Heatmap")
plt.show()
```

Darker red means a stronger positive relationship; darker blue means a stronger
negative relationship.

> Remember: a high correlation does **not** prove that one variable causes the
> other. It only says they move together in this sample.

## 9. Comparing Groups

We can compare the average final score by city:

```python
clean.groupby("city")["final_score"].mean().sort_values(ascending=False)
```

Then draw a bar chart:

```python
clean.groupby("city")["final_score"].mean().sort_values(ascending=False).plot(
    kind="bar", figsize=(7, 4)
)
plt.title("Average Final Score by City")
plt.xlabel("City")
plt.ylabel("Average final score")
plt.xticks(rotation=0)
plt.show()
```

Warning: each city has only a few students. A small difference may be pure
chance, so do not treat this chart as proof that one city is "better" than
another.

## 10. A Simple Conditional Probability

Let us calculate the probability that a student passed, given that they studied
at least 5 hours.

```
P(passed | hours_studied >= 5)
```

In code:

```python
studied_more = clean[clean["hours_studied"] >= 5]
probability = (studied_more["passed"] == "Yes").mean()

print("Students who studied >= 5 hours:", len(studied_more))
print("Probability of passing given >= 5 hours:", probability)
```

For this dataset, 10 students studied at least 5 hours and 9 of them passed, so
the probability is **0.9**.

Compare it with students who studied fewer than 5 hours:

```python
studied_less = clean[clean["hours_studied"] < 5]
print("Probability of passing given < 5 hours:",
      (studied_less["passed"] == "Yes").mean())
```

The probability is **0.0** in this small dataset. In a real dataset the numbers
would be less extreme, but the idea is the same: a condition can change the
probability a lot.

### Probability with a simulation

We can also estimate probability by repeating an experiment many times:

```python
np.random.seed(0)
rolls = np.random.randint(1, 7, size=10000)
print("Estimated P(rolling a 4):", (rolls == 4).mean())
```

The answer is close to `1/6 ≈ 0.1667`. This is called a **simulation**. More
repetitions usually give a more stable estimate.

## 11. Try It Yourself

Use the `clean` DataFrame to answer these questions:

1. What is the mean and median of `previous_score`?
2. How many students are from Pokhara, and how many passed?
3. What is the correlation between `previous_score` and `final_score`?
4. Calculate `P(passed | previous_score >= 60)`.
5. Which column has the largest standard deviation? Does that make sense given
   the units of each column?

Hints:

```python
clean["previous_score"].agg(["mean", "median"])
clean[clean["city"] == "Pokhara"]["passed"].value_counts()
clean[["previous_score", "final_score"]].corr()
high_previous = clean[clean["previous_score"] >= 60]
(high_previous["passed"] == "Yes").mean()
clean.describe().loc["std"].sort_values(ascending=False)
```

For question 5, remember that standard deviation is in the same units as the
data. Comparing the SD of `attendance_percent` (0–100) with the SD of
`hours_studied` (0–40) is not a fair comparison of "importance". Always look at
the units and the range.

## 12. What We Learned

In this practical we:

- Loaded a CSV file into a pandas DataFrame.
- Inspected rows, columns, data types, and missing values.
- Calculated mean, median, mode, and standard deviation.
- Drew histograms, box plots, scatter plots, and a correlation heatmap.
- Found an outlier using the IQR rule.
- Computed a conditional probability.

## Outcome

Learners can load a real dataset, summarize it with statistics, visualize its
distribution and relationships, detect basic data-quality issues, and compute a
simple conditional probability.
