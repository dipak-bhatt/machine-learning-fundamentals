# Day 09 — Practical: Decision Trees

**Module:** Module 2: Supervised Learning

In this practical we grow, read, and prune decision trees. We will:

- Compute entropy, Gini impurity, and information gain by hand.
- Train a decision tree classifier on the Iris data.
- Visualize the learned tree and read a single prediction's decision path.
- Watch training and test scores diverge as `max_depth` grows.
- Prune with `ccp_alpha` and compare validation scores.
- Read impurity-based feature importance.
- Train a decision tree regressor on the Day 04 house data.

The classification part uses the Iris data bundled with scikit-learn. The
regression part reuses `../../00-foundations/day-04-bias-variance-ml-workflow/house_prices.csv`,
so the notebook should be created inside this Day 09 folder.

## 1. Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    plot_tree,
    export_text,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
)

RANDOM_STATE = 42
```

`RANDOM_STATE` makes the split and the tree's tie-breaking repeatable. Trees can
produce different splits when two thresholds have exactly the same gain, so fixing
the seed keeps the expected outputs stable.

## 2. Warm-Up: Entropy and Gini by Hand

Impurity is the engine of the whole algorithm, so implement it once and check the
numbers. These functions assume a one-dimensional array of labels.

```python
def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))


def gini(labels):
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)


def weighted_entropy(*groups):
    total = sum(len(g) for g in groups)
    return sum(len(g) / total * entropy(g) for g in groups)
```

A balanced two-class group should give entropy 1 and Gini 0.5:

```python
parent = np.array(["sweet"] * 5 + ["not"] * 5)

print("entropy:", round(entropy(parent), 3))
print("gini:", round(gini(parent), 3))
```

Expected output:

```text
entropy: 1.0
gini: 0.5
```

Now score the two candidate splits from the theory section. The first question is
colour, the second is size.

```python
colour_red = np.array(["sweet"] * 4 + ["not"] * 1)
colour_green = np.array(["sweet"] * 1 + ["not"] * 4)

size_big = np.array(["sweet"] * 3 + ["not"] * 2)
size_small = np.array(["sweet"] * 2 + ["not"] * 3)

for name, a, b in [
    ("colour", colour_red, colour_green),
    ("size", size_big, size_small),
]:
    child_h = weighted_entropy(a, b)
    gain = entropy(parent) - child_h
    print(f"{name}: child entropy={child_h:.3f} gain={gain:.3f}")
```

Expected output:

```text
colour: child entropy=0.722 gain=0.278
size: child entropy=0.971 gain=0.029
```

Colour wins. The tree would ask about colour first, even though both questions
produce two equally sized children. The difference is purity, not balance.

As a sanity check, compute the Gini reduction for the colour split:

```python
colour_gini = (5 / 10) * gini(colour_red) + (5 / 10) * gini(colour_green)
print("gini reduction:", round(gini(parent) - colour_gini, 3))
```

Expected output:

```text
gini reduction: 0.18
```

Entropy gain and Gini reduction are both positive and both prefer colour, but they
are not on the same numeric scale. Never compare the two values directly.

## 3. Load and Inspect the Iris Data

Iris has 150 flowers, four numeric features, and three species. It is small,
balanced, and well understood, which makes it ideal for reading a tree.

```python
iris = load_iris(as_frame=True)

df = iris.frame
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

print(df.shape)
print(df["species"].value_counts())
df.head()
```

Expected output:

```text
(150, 6)
species
setosa        50
versicolor    50
virginica     50
Name: count, dtype: int64
```

The features are all measured in centimetres, but a tree does not need them
scaled. Splits compare one feature to a threshold, so units do not matter. That is
a real convenience compared with logistic regression and neural networks.

## 4. Split the Data First

As always, split before looking at the test set. `stratify=y` keeps the three
species balanced in both parts.

```python
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=RANDOM_STATE,
)

print(X_train.shape, X_test.shape)
```

Expected output:

```text
(105, 4) (45, 4)
```

## 5. Establish a Baseline

Before training anything, ask what a trivial rule achieves. With three balanced
classes, a model that always predicts one class gets about one third.

```python
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
print("baseline accuracy:", round(baseline_acc, 3))
```

Expected output:

```text
baseline accuracy: 0.333
```

Any real model must beat this clearly. On Iris it will, easily, but the habit
matters more than the number.

## 6. Train a Full Tree

First let the tree grow with no limits. This is deliberate: we want to see what
overfitting looks like before we fix it.

```python
full_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
full_tree.fit(X_train, y_train)

print("depth:", full_tree.get_depth())
print("leaves:", full_tree.get_n_leaves())
print("train accuracy:", round(full_tree.score(X_train, y_train), 3))
print("test accuracy:", round(full_tree.score(X_test, y_test), 3))
```

Expected output:

```text
depth: 5
leaves: 8
train accuracy: 1.0
test accuracy: 0.933
```

The training accuracy is perfect. With only 105 rows and a modest feature set, the
tree can carve the training points into pure leaves. Test accuracy is lower, which
is normal, but note that the gap is not catastrophic here. Iris is a clean dataset;
on noisier data the gap would be much larger.

Look at the confusion matrix:

```python
y_pred = full_tree.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

Expected output:

```text
[[15  0  0]
 [ 0 12  3]
 [ 0  0 15]]
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        15
  versicolor       1.00      0.80      0.89        15
   virginica       0.83      1.00      0.91        15

    accuracy                           0.93        45
   macro avg       0.94      0.93      0.93        45
weighted avg       0.94      0.93      0.93        45
```

The mistakes go in one direction: three versicolor flowers are predicted as
virginica. Those two species overlap in petal measurements, so the tree pushes the
boundary toward virginica. Setosa is far from the others and is never
misclassified. Note that the majority-class baseline (33%) and this tree (93%)
differ by a wide margin, which is what we want to see.

## 7. Visualize the Learned Tree

A tree is one of the few models we can literally draw.

```python
plt.figure(figsize=(14, 8))
plot_tree(
    full_tree,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=9,
)
plt.show()
```

Each box shows:

- the question (or `leaf` for a terminal node),
- the Gini impurity at that node,
- the number of training `samples` that reached it,
- the `value` counts per class,
- the `class` the node would predict.

`filled=True` colours boxes by their majority class. `rounded=True` softens the
corners. If the text is cramped, reduce `fontsize` or export the tree as text.

For a compact view, `export_text` prints the rules directly:

```python
small_tree = DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE)
small_tree.fit(X_train, y_train)

print(export_text(small_tree, feature_names=list(iris.feature_names)))
```

Expected output:

```text
|--- petal length (cm) <= 2.45
|   |--- class: 0
|--- petal length (cm) >  2.45
|   |--- petal width (cm) <= 1.55
|   |   |--- class: 1
|   |--- petal width (cm) >  1.55
|   |   |--- class: 2
```

This is the entire model. If a flower's petal length is at most 2.45 cm, the tree
says setosa. Otherwise it checks petal width, splitting versicolor from virginica
at 1.55 cm. The whole classifier fits in three rules.

## 8. Read a Decision Path

A prediction is the path the row takes. We can extract that path exactly.

```python
row = X_test.iloc[[0]]

print("true:", iris.target_names[y_test.iloc[0]])
print("predicted:", iris.target_names[full_tree.predict(row)[0]])
print("probabilities:", np.round(full_tree.predict_proba(row)[0], 3))
```

Expected output:

```text
true: virginica
predicted: virginica
probabilities: [0. 0. 1.]
```

Now walk the nodes:

```python
node_indicator = full_tree.decision_path(row)

for node in node_indicator.indices:
    feature = full_tree.tree_.feature[node]
    threshold = full_tree.tree_.threshold[node]

    if feature >= 0:
        name = iris.feature_names[feature]
        value = row.iloc[0, feature]
        direction = "left" if value <= threshold else "right"
        print(f"node {node}: {name} = {value:.1f} <= {threshold:.2f}? {direction}")
    else:
        print(f"node {node}: leaf, class counts {full_tree.tree_.value[node]}")
```

Expected output:

```text
node 0: petal length (cm) = 6.3 <= 2.45? right
node 2: petal width (cm) = 1.8 <= 1.55? right
node 6: petal width (cm) = 1.8 <= 1.70? right
node 10: petal length (cm) = 6.3 <= 4.85? right
node 14: leaf, class counts [[0. 0. 1.]]
```

The path has five nodes and uses only two features. Sepal length and sepal width
never appear. This is the explanation a user would receive: four tests, then a leaf
that contains exactly one training flower, a virginica. That purity is why
`predict_proba` returns `[0, 0, 1]`.

A leaf with a single sample is a good reminder of the overfitting risk. The path is
easy to read, but the leaf's certainty comes from one observation. Deeper trees
produce more paths like this one, and those paths are exactly the ones that fail to
generalize. Always inspect the `samples` and `value` counts, not just the predicted
class.

## 9. Control Overfitting with `max_depth`

Now vary the depth and record training and test accuracy. This is the single most
useful experiment with a tree.

```python
results = []

for depth in [1, 2, 3, 4, 5, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)
    results.append(
        {
            "max_depth": depth,
            "actual_depth": tree.get_depth(),
            "leaves": tree.get_n_leaves(),
            "train": tree.score(X_train, y_train),
            "test": tree.score(X_test, y_test),
        }
    )

results_df = pd.DataFrame(results).round(3)
print(results_df.to_string(index=False))
```

Expected output:

```text
 max_depth  actual_depth  leaves  train   test
       1.0             1       2  0.667  0.667
       2.0             2       3  0.971  0.889
       3.0             3       5  0.981  0.978
       4.0             4       7  0.990  0.889
       5.0             5       8  1.000  0.933
       NaN             5       8  1.000  0.933
```

Read the table as a story:

- Depth 1 is too simple. Two leaves cannot separate three species.
- Depth 3 is the sweet spot on this split: 97.8% test accuracy with only five
  leaves.
- Depth 4 and beyond add leaves and training accuracy, but test accuracy does not
  improve. Those extra branches are fitting details of the training set.

A single split can be lucky or unlucky. Use cross-validation to check whether the
pattern is stable:

```python
for depth in [1, 2, 3, 4, 5, None]:
    scores = cross_val_score(
        DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE),
        X,
        y,
        cv=5,
    )
    print(f"depth={depth}: mean={scores.mean():.3f}")
```

Expected output:

```text
depth=1: mean=0.667
depth=2: mean=0.933
depth=3: mean=0.973
depth=4: mean=0.953
depth=5: mean=0.953
depth=None: mean=0.953
```

Cross-validation confirms the shape: depth 1 is too simple, depth 3 is the peak,
and deeper trees fall back slightly. The exact winner is less important than the
lesson: past a point, more depth does not buy generalization.

## 10. Prune with Cost-Complexity

`max_depth` is a blunt instrument. **Cost-complexity pruning** instead grows a full
tree and removes weak branches, trading training error for a smaller tree. The
strength of the penalty is `ccp_alpha`.

First see which alphas are meaningful for this data:

```python
path = DecisionTreeClassifier(random_state=RANDOM_STATE).cost_complexity_pruning_path(
    X_train, y_train
)
print(np.round(path.ccp_alphas, 4))
```

Expected output:

```text
[0.     0.009  0.0185 0.2789 0.3333]
```

Now evaluate a few values with cross-validation on the training set:

```python
alphas = [0.0, 0.005, 0.01, 0.02, 0.05]

for alpha in alphas:
    scores = cross_val_score(
        DecisionTreeClassifier(ccp_alpha=alpha, random_state=RANDOM_STATE),
        X_train,
        y_train,
        cv=5,
    )
    tree = DecisionTreeClassifier(ccp_alpha=alpha, random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)
    print(
        f"alpha={alpha}: depth={tree.get_depth()} "
        f"leaves={tree.get_n_leaves()} cv={scores.mean():.3f}"
    )
```

Expected output:

```text
alpha=0.0: depth=5 leaves=8 cv=0.943
alpha=0.005: depth=5 leaves=8 cv=0.943
alpha=0.01: depth=3 leaves=4 cv=0.943
alpha=0.02: depth=2 leaves=3 cv=0.952
alpha=0.05: depth=2 leaves=3 cv=0.943
```

A small `alpha` such as 0.01 removes branches without hurting validation accuracy.
In this dataset the tree is already small, so pruning mainly simplifies it. On
larger, noisier data the effect is more dramatic.

Finally, fit the pruned tree and score it once on the test set:

```python
pruned_tree = DecisionTreeClassifier(ccp_alpha=0.01, random_state=RANDOM_STATE)
pruned_tree.fit(X_train, y_train)

print("depth:", pruned_tree.get_depth())
print("leaves:", pruned_tree.get_n_leaves())
print("train accuracy:", round(pruned_tree.score(X_train, y_train), 3))
print("test accuracy:", round(pruned_tree.score(X_test, y_test), 3))
```

Expected output:

```text
depth: 3
leaves: 4
train accuracy: 0.981
test accuracy: 0.933
```

The pruned tree keeps test accuracy while cutting the number of leaves from eight
to four. That is the trade we want: the same honest performance with a simpler,
more defensible model.

## 11. Feature Importance

Every split reduces impurity weighted by the rows that reach it. Summing per
feature gives `feature_importances_`.

```python
importance = pd.Series(
    full_tree.feature_importances_,
    index=iris.feature_names,
).sort_values(ascending=False)

print(importance.round(3))
```

Expected output:

```text
petal length (cm)    0.541
petal width (cm)     0.430
sepal width (cm)     0.029
sepal length (cm)    0.000
```

The two petal measurements do almost all the work. That matches the rules we read
earlier: every split was on petal length or petal width. Sepal length is never used
in this particular tree.

Two cautions:

- The importances are specific to this tree. A different random tie-break can shift
  credit between correlated features.
- Impurity-based importance can favour features with many distinct values. For a
  more robust ranking, use `sklearn.inspection.permutation_importance` on a
  validation set.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    full_tree,
    X_test,
    y_test,
    n_repeats=10,
    random_state=RANDOM_STATE,
)

perm = pd.Series(result.importances_mean, index=iris.feature_names)
print(perm.sort_values(ascending=False).round(3))
```

Expected output (values may vary slightly with library version):

```text
petal length (cm)    0.507
petal width (cm)     0.089
sepal width (cm)     0.007
sepal length (cm)    0.000
```

Permutation importance agrees that the petal features matter most, and it assigns
almost zero to the sepal features because shuffling them barely hurts this tree.
The two methods rank the features the same way, which makes the conclusion
stronger. Exact values vary with library version and the random repeats, so treat
the ranking as the result, not the decimals.

## 12. A Regression Tree on House Prices

Classification trees choose splits by impurity. Regression trees choose splits
that reduce squared error, and each leaf predicts the mean price of its training
rows. Let us apply one to the Day 04 house data.

```python
df = pd.read_csv(
    "../../00-foundations/day-04-bias-variance-ml-workflow/house_prices.csv"
)

df = df.dropna(subset=["age_years"]).copy()
df = pd.get_dummies(df, columns=["location"], drop_first=True)

features = [c for c in df.columns if c != "price_usd"]
X_house = df[features]
y_house = df["price_usd"]

Xh_train, Xh_test, yh_train, yh_test = train_test_split(
    X_house,
    y_house,
    test_size=0.25,
    random_state=RANDOM_STATE,
)

print(features)
print(Xh_train.shape, Xh_test.shape)
```

Expected output:

```text
['area_sqft', 'age_years', 'location_Suburb', 'location_Town']
(100, 4) (34, 4)
```

Compare a linear baseline with trees of different depths:

```python
linear = LinearRegression()
linear.fit(Xh_train, yh_train)

print(
    "linear train RMSE:",
    round(np.sqrt(mean_squared_error(yh_train, linear.predict(Xh_train))), 1),
    "test RMSE:",
    round(np.sqrt(mean_squared_error(yh_test, linear.predict(Xh_test))), 1),
)

for depth in [2, 3, 4, None]:
    reg_tree = DecisionTreeRegressor(max_depth=depth, random_state=RANDOM_STATE)
    reg_tree.fit(Xh_train, yh_train)
    train_rmse = np.sqrt(mean_squared_error(yh_train, reg_tree.predict(Xh_train)))
    test_rmse = np.sqrt(mean_squared_error(yh_test, reg_tree.predict(Xh_test)))
    print(
        f"depth={depth}: train RMSE={train_rmse:.1f} "
        f"test RMSE={test_rmse:.1f} leaves={reg_tree.get_n_leaves()}"
    )
```

Expected output:

```text
linear train RMSE: 11058.1 test RMSE: 12427.1
depth=2: train RMSE=14375.7 test RMSE=24044.4 leaves=4
depth=3: train RMSE=10087.8 test RMSE=15429.4 leaves=8
depth=4: train RMSE=7364.5 test RMSE=13396.5 leaves=16
depth=None: train RMSE=0.0 test RMSE=13866.5 leaves=96
```

Several lessons appear at once:

- The linear model is a strong baseline here. The relationship between area and
  price is roughly linear.
- The shallow tree underfits; four leaves are not enough.
- The full tree drives training error to exactly zero, one leaf per training row,
  and still loses to the linear model on test data.
- A depth-4 tree is competitive but not better than the line.

This is an honest result: trees are not automatically superior. They shine when
the data contain threshold effects and interactions, and this synthetic dataset is
mostly a straight line. The exercise of comparing them is the point.

## 13. Try It Yourself

1. **Change the criterion.** Fit the Iris tree with `criterion="entropy"` and
   compare depth, leaves, and test accuracy with the Gini version.
2. **Grow a stump.** Set `max_depth=1` and draw it. What single question does the
   tree consider most valuable across all three species?
3. **Force a larger leaf.** Try `min_samples_leaf=20` on the house data and explain
   why the tree becomes shallower.
4. **Break the stability.** Refit the Iris tree with three different
   `random_state` values and compare the root split. Does it always choose petal
   length?
5. **Add a leaky feature.** Append a column of random numbers to `X_train` and
   `X_test`, refit, and inspect `feature_importances_`. Why can a useless feature
   receive non-zero importance?
6. **Explain a rejection.** Pick a test row from the house data and print its full
   decision path. Translate the path into a sentence a customer could understand.

## 14. What We Learned

- Entropy and Gini measure how mixed a node is; information gain is the impurity
  drop from a split.
- The tree grows greedily, choosing the best split at each node, and leaf
  predictions are the majority class or mean target.
- A fully grown tree can reach perfect training accuracy and then overfit.
- `max_depth`, `min_samples_leaf`, and `ccp_alpha` control capacity; validation
  data, not training scores, choose among them.
- A decision path is a complete, human-readable explanation of one prediction.
- Feature importance is a useful ranking but is biased and tree-specific.
- Trees handle nonlinearity and interactions without scaling, but they are unstable
  and cannot extrapolate. On the house data, a linear model remained stronger.

## Outcome

Learners can compute entropy, Gini impurity, and information gain, train and
visualize a decision tree, read and explain a decision path, choose depth and
pruning settings using validation data, interpret feature importance, and compare a
tree fair-and-square against a linear baseline on both classification and
regression tasks.
