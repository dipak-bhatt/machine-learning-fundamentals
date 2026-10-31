# Day 02 — Theory: Data, Statistics, and Probability

**Module:** Module 1: Foundations

Yesterday we learned that machine learning (ML) finds patterns in examples.
Today we look at the examples themselves: **data**. We will learn how a dataset
is organized, how to summarize it with simple statistics, and how probability
helps us reason about uncertainty.

No advanced math is needed. Every idea below is explained with a small, everyday
example.

## 1. The Building Blocks of a Dataset

A **dataset** is simply a table of information. Imagine we collected data about
students:

| student_id | hours_studied | attendance_percent | final_score | city      | passed |
|-----------:|--------------:|-------------------:|------------:|-----------|--------|
| 1          | 2             | 60                 | 48          | Kathmandu | No     |
| 2          | 3             | 65                 | 52          | Kathmandu | No     |
| 3          | 5             | 75                 | 65          | Pokhara   | Yes    |
| 4          | 6             | 80                 | 70          | Pokhara   | Yes    |

Here is the vocabulary:

- **Dataset:** the whole table.
- **Row (also called a sample, example, observation, or record):** one student.
- **Column (also called a feature, variable, or attribute):** one piece of
  information, such as `hours_studied`.
- **Feature:** a column we use as input to make a prediction.
- **Target (also called label, output, or dependent variable):** the column we
  want to predict.

If we want to predict `final_score`, then:

- **Features (X):** `hours_studied`, `attendance_percent`, `previous_score`.
- **Target (y):** `final_score`.

```
X (inputs)  →  model  →  y (prediction)
```

The same column can be a feature in one problem and a target in another. If we
want to predict whether a student passes, then `passed` is the target.

> Tip: Different books use different words for the same idea. "Feature",
> "input", and "predictor" usually mean the same thing. "Target", "label", and
> "output" usually mean the same thing too.

## 2. Types of Data

Not all data is the same. Knowing the type tells us what we can calculate and
how we should prepare the data for a model.

### Numerical data (numbers)

- **Continuous:** can take any value in a range.
  - Examples: height (172.5 cm), price (₹299.99), temperature (36.6 °C).
- **Discrete:** whole numbers you can count.
  - Examples: number of children (0, 1, 2), number of rooms (3), number of
    clicks (125).

### Categorical data (groups or labels)

- **Nominal:** categories with no order.
  - Examples: city (Kathmandu, Pokhara), color (red, blue), blood group (A, B, O).
- **Ordinal:** categories with a meaningful order.
  - Examples: small / medium / large, poor / average / good, ratings 1–5 stars.

### Other common types

- **Text:** reviews, emails, tweets.
- **Dates and times:** order dates, signup date, transaction time.
- **Images, audio, video:** special data that needs special models.

### Why data types matter

1. **Statistics:** a mean makes sense for `final_score` (a number), but not for
   `city` (a label). For `city`, counting each category is more useful.
2. **Charts:** numbers are often shown with histograms or scatter plots;
   categories are shown with bar charts.
3. **Models:** most ML models need numbers. Categorical values like
   "Kathmandu" and "Pokhara" must be converted to numbers first (for example,
   using one-hot encoding). We will see this in later days.
4. **Scale:** `attendance_percent` (0–100) and `hours_studied` (0–10) are on very
   different scales. Sometimes we rescale them so no feature dominates.

## 3. Describing Data with Statistics

Statistics are short summaries of a large amount of data. Instead of reading
1,000 exam scores, we can describe them with a few numbers.

### Mean (average)

Add all values and divide by how many there are.

Example: exam scores `2, 4, 6, 8`

```
mean = (2 + 4 + 6 + 8) / 4 = 20 / 4 = 5
```

The mean is easy to understand, but it is **sensitive to extreme values**.

Salary example: five people earn `20,000, 22,000, 25,000, 28,000, 1,00,000`.

```
mean = 39,000
```

Nobody actually earns 39,000. The single very high salary pulls the average up.
In situations like this, the median is a better summary.

### Median (middle value)

Sort the values from smallest to largest and take the middle one.

- Odd number of values: `2, 4, 6, 8, 10` → median = **6**.
- Even number of values: `2, 4, 6, 8` → middle two are 4 and 6 → median =
  **(4 + 6) / 2 = 5**.

For the salary example, the sorted values are
`20,000, 22,000, 25,000, 28,000, 1,00,000`, so the median is **25,000** — much
closer to what a typical person earns.

### Mode (most frequent value)

The value that appears most often.

Example: shoe sizes `7, 8, 8, 9, 10` → mode = **8**.

Mode is especially useful for categorical data. For `city`, the mode is the city
that appears most often in the dataset.

### Which one should I use?

| Statistic | Best used when | Example |
|-----------|----------------|---------|
| Mean | Data is fairly balanced, no big outliers | Average height of a class |
| Median | Data has outliers or is skewed | House prices, salaries |
| Mode | Data is categorical, or you want the most common value | Most common city |

### Range, variance, and standard deviation

The mean tells us the center, but not how spread out the data is.

- **Range** = maximum − minimum.
  - Scores `50, 60, 70, 80` → range = 80 − 50 = **30**.
- **Variance** = the average squared distance from the mean.
  - It measures spread, but its units are squared (hard to interpret directly).
- **Standard deviation (SD)** = the square root of the variance.
  - It is in the same units as the data, so it is easier to understand.

A small standard deviation means values are close to the mean. A large standard
deviation means values are spread out.

Example:

- Class A scores: `68, 70, 72, 70, 70` → mean ≈ 70, SD is small.
- Class B scores: `50, 60, 70, 80, 90` → mean = 70, SD is large.

Both classes have the same average, but Class B is much more spread out. This is
why we should not look at the mean alone.

### Percentiles and quartiles (a quick look)

A **percentile** tells you the value below which a given percentage of data
falls.

- The **50th percentile** is the median.
- The **25th percentile** is called **Q1**.
- The **75th percentile** is called **Q3**.
- **Interquartile range (IQR)** = Q3 − Q1.

The IQR describes the middle 50% of the data. It is also a common way to detect
outliers, which we will use in the practical.

## 4. Distributions: Seeing the Shape of Data

A **distribution** shows how often each value appears. The easiest way to see it
is a **histogram**: values on the x-axis, and how many times they appear on the
y-axis.

Common shapes:

- **Normal (bell-shaped):** most values near the middle, fewer at the extremes.
  - Example: heights of adult people, exam scores in a large class.
- **Right-skewed (tail to the right):** a few very large values.
  - Example: income, house prices, number of followers.
- **Left-skewed (tail to the left):** a few very small values.
  - Example: scores on a very easy test where most students score high.
- **Uniform:** all values are roughly equally common.
  - Example: rolling a fair six-sided die.
- **Bimodal:** two peaks, suggesting two groups.
  - Example: customer ages when a shop serves both students and retired people.

Why it matters: the shape tells us whether the mean is a good summary. For
skewed data, the median is often safer. It also helps us spot data-entry
mistakes, such as impossible ages.

## 5. Population vs Sample

- **Population:** every item we care about.
  - Example: all 10,000 customers of a company.
- **Sample:** a smaller part of the population that we actually measure.
  - Example: 500 randomly chosen customers.

We usually cannot measure everyone, so we use a sample and **estimate** the
population.

```
Population (all 10,000 customers)
        ↓ take a sample
Sample (500 customers) → calculate mean → estimate for the population
```

For the estimate to be trustworthy, the sample must be **representative**. A
sample of only morning shoppers may not represent evening shoppers. This kind of
problem is called **sampling bias**.

In ML, the data we train on is also just a sample. If that sample is biased, the
model's predictions will be biased too.

## 6. Correlation vs Causation

**Correlation** means two variables tend to move together.

- **Positive correlation:** when one goes up, the other tends to go up.
  - Example: hours studied and exam score.
- **Negative correlation:** when one goes up, the other tends to go down.
  - Example: hours spent watching TV and exam score.
- **No correlation:** no clear pattern.
  - Example: shoe size and exam score.

The **correlation coefficient (r)** is a number between −1 and +1:

- `r = 1` → perfect positive relationship.
- `r = 0` → no linear relationship.
- `r = −1` → perfect negative relationship.
- Values near 0 (for example, 0.1 or −0.2) mean a weak relationship.

**Causation** means one thing actually causes the other.

A classic example: ice cream sales and drowning incidents both increase in
summer. Does ice cream cause drowning? No. A third factor — hot weather — causes
both more ice cream sales and more swimming. This hidden third factor is called
a **confounding variable** (or lurking variable).

```
Hot weather
   ├──→ more ice cream sales
   └──→ more swimming → more drownings
```

**Correlation is not causation.** ML models learn patterns and correlations.
They do not automatically know what causes what. Be careful before using a model
to make important decisions.

## 7. Probability Basics

**Probability** measures how likely an event is. It is always between 0 and 1:

- `0` → impossible.
- `1` → certain.
- `0.5` → equally likely to happen or not.

It can also be written as a percentage: `0.5 = 50%`.

### Simple examples

- A fair coin: `P(heads) = 1/2 = 0.5`.
- A fair six-sided die: `P(rolling a 4) = 1/6 ≈ 0.167`.
- A standard deck of 52 cards: `P(drawing an ace) = 4/52 = 1/13 ≈ 0.077`.

General idea:

```
P(event) = number of favourable outcomes / total number of outcomes
```

The **complement** rule: the probability that an event does **not** happen is
`1 − P(event)`.

Example: if `P(rain) = 0.3`, then `P(no rain) = 1 − 0.3 = 0.7`.

### Independent events

Two events are **independent** if one does not affect the other. For independent
events, multiply the probabilities:

```
P(two heads) = P(heads) × P(heads) = 1/2 × 1/2 = 1/4 = 0.25
```

### Conditional probability

**Conditional probability** is the probability of an event **given** that
another event has already happened. We write it as `P(A | B)`, read as
"probability of A given B".

Example with 100 emails:

- 40 emails are spam.
- 30 emails contain the word "free".
- 24 emails are spam **and** contain "free".

Then:

- `P(spam | contains "free") = 24 / 30 = 0.8`
  - Among emails with "free", 80% are spam.
- `P(contains "free" | spam) = 24 / 40 = 0.6`
  - Among spam emails, 60% contain "free".

Notice these two numbers are different. The order matters.

The formula is:

```
P(A | B) = P(A and B) / P(B)
```

### A first look at Bayes' theorem

Bayes' theorem lets us **update** a belief when we see new evidence:

```
P(A | B) = P(B | A) × P(A) / P(B)
```

In plain words:

- `P(A)` is our **prior** belief before seeing evidence.
- `P(B | A)` is how likely the evidence is if A is true.
- `P(A | B)` is our **posterior** belief after seeing the evidence.

Spam example: most email is not spam, so our prior `P(spam)` is low. But if an
email contains "free money", that evidence is much more common in spam, so the
posterior probability of spam rises. We will study this properly on Day 12.

### Why probability matters in ML

Most ML models do not just say "spam" or "not spam". They output a probability,
such as **0.92 spam**. That number tells us how confident the model is, and it
helps us decide how to act.

## 8. Messy Data: Missing Values, Outliers, and Noise

Real datasets are rarely clean. Before training a model, we must look for
problems.

### Missing values

A missing value is a blank cell. It may happen because:

- the information was never collected,
- a sensor failed,
- a customer skipped a question,
- data was lost during transfer.

Options for handling missing values:

1. **Remove** the row or column (only if very few values are missing).
2. **Fill** with a sensible value, such as the mean, median, or mode.
3. **Predict** the missing value using other columns.
4. Use a model that can handle missing values directly.

There is no single correct choice. Always ask: *why* is the value missing? If
high-income customers never report their income, filling with the average may
mislead the model.

### Outliers

An **outlier** is a value far away from the rest.

- Example: ages `22, 25, 27, 30, 95`. The `95` is an outlier.
- It could be a **data-entry error** (someone typed 95 instead of 29).
- It could be a **real but rare value** (a very wealthy customer).

Outliers can pull the mean and change a model. Do not delete them
automatically. First investigate whether they are mistakes or genuine.

### Noise

**Noise** is random error in the data.

- A typo: age entered as `220` instead of `22`.
- A shaky sensor giving slightly different readings each time.
- A customer clicking the wrong rating by accident.

Noise makes patterns harder to see. Cleaning, validating, and collecting more
data can reduce its effect.

### Other common problems

- **Duplicates:** the same customer appears twice, so the model counts them
  twice.
- **Inconsistent labels:** `"NY"`, `"New York"`, and `"new york"` mean the same
  place but are treated as three categories.
- **Wrong types:** a number stored as text, such as `"25"` instead of `25`.

A simple rule: **garbage in, garbage out**. Good ML starts with good data.

## 9. How Today's Topics Fit Together

Before training a model, we:

1. Understand what each **row** and **column** means.
2. Identify **feature types** (numerical or categorical).
3. Summarize the data with **mean, median, mode, and standard deviation**.
4. Look at **distributions** to understand shape and spread.
5. Remember that our data is a **sample**, not the whole population.
6. Use **correlation** to find relationships, without assuming causation.
7. Use **probability** to reason about uncertainty.
8. Handle **missing values, outliers, and noise**.

## 10. ML Use Cases for Today's Concepts

Each concept from today is not just theory. Here is how it appears in real
machine-learning work.

| Theory concept | How it is used in ML | Real use case |
|----------------|----------------------|---------------|
| Rows, features, and target | Defines the inputs `X` and the answer `y` that the model learns to predict | House price prediction: area and bedrooms are features; price is the target |
| Numerical vs categorical data | Numerical columns may be scaled; categorical columns must be encoded as numbers | Customer churn: `monthly_charge` is scaled, while `contract_type` is one-hot encoded |
| Mean, median, mode | Fill missing values, summarize data, and set a simple baseline | Filling missing income with the median because income is skewed |
| Range, variance, standard deviation | Detect unusual values, compare spread, and scale features | Fraud detection: a transaction amount far from the normal range deserves a look |
| Distributions | Choose a model or transformation and spot data problems | House prices are right-skewed, so a log transform or a tree model may help |
| Population vs sample | Split data into training and test sets and judge whether the model generalizes | Train on last year's customers, test on this year's customers |
| Correlation vs causation | Select useful features and avoid trusting misleading patterns | Support calls correlate with churn, but the calls may be a *result* of churn, not its cause |
| Probability | Produce confidence scores, choose decision thresholds, and apply Bayes' rule | A spam filter outputs "0.92 probability of spam" |
| Missing values, outliers, and noise | Clean data, use robust statistics, and detect anomalies | A typo of age `220` or a failed sensor reading must be handled before training |

### Use case 1: Predicting house prices

A real-estate company wants to predict the selling price of a house.

- **Row:** one house.
- **Features:** area, number of bedrooms, location, age of the house.
- **Target:** selling price.
- **Data types:** area is numerical; location is categorical, so it needs
  encoding.
- **Statistics:** the median price is more useful than the mean because a few
  very expensive houses pull the mean upward.
- **Distribution:** house prices are usually right-skewed.
- **Outliers:** a mansion may be a genuine outlier, not an error, so we should
  not delete it automatically.
- **Sample:** we train on past sales and test on recent sales that the model has
  never seen.
- **Correlation:** larger area is correlated with higher price. Other factors,
  such as location and condition, also affect price, so area is not the only
  cause.
- **Probability:** instead of one exact price, the company may want a range or
  the probability that a house sells above a target price.

### Use case 2: A spam filter

An email provider wants to classify each email as spam or not spam.

- **Row:** one email.
- **Features:** words in the subject, sender reputation, number of links.
- **Target:** spam or not spam.
- **Probability:** the model outputs a score such as `P(spam) = 0.92`.
  A threshold (for example, 0.5) turns that score into a decision.
- **Conditional probability:** `P(spam | contains "free")` is the chance an
  email is spam given that it contains the word "free". This is the idea behind
  Naive Bayes (Day 12).
- **Threshold trade-off:** lowering the threshold catches more spam but may also
  mark some normal emails as spam.
- **Data quality:** duplicate emails, missing sender information, and
  incorrectly labelled emails all add noise.
- **Sampling bias:** if the training emails all come from one email provider,
  the filter may work poorly for other providers.

### Use case 3: Predicting customer churn

A company wants to know which customers are about to leave (churn).

- **Row:** one customer.
- **Features:** tenure, monthly charge, contract type, number of support calls.
- **Target:** churned or stayed.
- **Data types:** `contract_type` is categorical; `monthly_charge` is
  numerical.
- **Missing values:** some customers may not have a recorded income. Filling it
  with the median is one option, but we should ask why it is missing.
- **Class imbalance:** usually far more customers stay than leave. A model that
  always predicts "stays" can look accurate while being useless, which is why we
  need better metrics (Day 18).
- **Correlation vs causation:** customers who call support often are more likely
  to churn. But the calls may be caused by the same problem that causes churn.
  Reducing calls will not automatically reduce churn.
- **Probability:** the model gives each customer a churn probability. The company
  can target the highest-risk customers with retention offers.
- **Threshold:** if an offer is expensive, the company may contact only customers
  above a high probability, such as 0.8.

### A quick checklist for any ML problem

Before training, ask:

1. What is one row, and what is the target?
2. Which columns are numbers and which are categories?
3. Are the values balanced, skewed, or spread out?
4. Is this data a fair sample of the real world?
5. Is any relationship just correlation, or is it causal?
6. What probabilities or confidence scores will the model produce?
7. What is missing, duplicated, noisy, or unusual?

## Outcome

Learners can describe the structure of a dataset, identify data types, summarize
data with basic statistics, read simple distributions, explain correlation and
conditional probability, and recognize common data-quality problems.
