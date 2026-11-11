# Day 09 — Theory: Decision Trees

**Module:** Module 2: Supervised Learning

Every model so far has made one big assumption: the prediction is a **linear
combination** of the features, or of features we engineered by hand. Logistic
regression draws a line; a polynomial adds curves; a neural network stacks linear
pieces. All of them output one smooth formula, and if the true pattern is a set of
"if this, then that" rules, that formula can be awkward.

A **decision tree** takes a different route. It asks a sequence of yes/no questions
and ends at an answer. "Is the petal length below 2.45 cm? If yes, setosa. If no,
is the petal width below 1.55 cm? ..." There is no equation to read; there is a
flowchart. That makes trees easy to explain, easy to draw, and surprisingly
effective on tabular data. It also makes them easy to overfit and unstable, which
is why we will spend as much time on pruning as on growing them.

Today we build the tree from first principles: how it represents knowledge, which
split it should choose, how impurity scores that choice, how the greedy learning
algorithm grows the tree, and how to stop it from memorizing noise.

## In Plain English

**Think of it like this:** a decision tree is a flowchart of yes/no questions
that ends in a prediction.

- Each question **splits** the data to make the groups purer (using Gini or
  entropy).
- Trees capture nonlinear patterns and interactions automatically and do not
  need feature scaling.
- Deep trees memorize noise, so **prune** them with limits on depth or leaf
  size.
- One tree is unstable; **ensembles** of trees fix that.
- Trees cannot predict beyond the target values they saw in training.

## 1. From Linear Models to Trees

A linear model assumes the target changes by a fixed amount when a feature
increases by one unit, no matter where we are. A tree assumes the target is
**piecewise constant**: within a region of feature space, all rows get the same
prediction. The tree finds those regions by cutting the space into boxes.

| Aspect | Linear / logistic regression | Decision tree |
|--------|------------------------------|---------------|
| Shape of the model | one global formula | a set of rules and regions |
| Decision boundary | a line, plane, or curve we design | axis-aligned rectangles |
| Feature scaling | often important | irrelevant; splits are on values |
| Feature interactions | must be added by hand | discovered automatically |
| Interpretability | coefficients | a drawable flowchart |
| Stability | stable | sensitive to small data changes |
| Extrapolation | continues the line | cannot go beyond seen targets |

The last row is a warning. A tree can only predict values it has seen in its
leaves. It cannot extrapolate a trend the way a line can. That is fine for many
classification tasks and a real limitation for forecasting.

None of this makes trees "better" than linear models. It makes them a different
kind of tool, and the comparison itself is a lesson: the best model depends on the
shape of the problem, not on a leaderboard.

## 2. The Parts of a Tree

A decision tree is made of **nodes** and **branches**:

- The **root node** is the first question. Every prediction starts there.
- An **internal node** (also called a split node) asks a yes/no question about one
  feature and sends each row to a child.
- A **branch** is the answer to that question, "yes" or "no", "left" or "right".
- A **leaf** (also called a terminal node) has no question. It stores the final
  prediction.

Here is a small tree for deciding whether to play tennis, based on a simplified
version of the classic weather dataset (the "Overcast" branch is omitted for
brevity):

```text
                 Outlook?
                /        \
          Sunny           Rain
            |               |
      Humidity?          Play = yes
       /      \
   High        Normal
     |            |
Play = no     Play = yes
```

The questions are usually of the form "is feature `j` less than or equal to a
threshold `t`?" This is called a **binary split**. For a categorical feature, the
question can instead be "is the value in this subset?", but scikit-learn's tree
implementations expect numeric input, so categories are encoded first.

Each leaf holds a **prediction rule**:

- For classification, the leaf stores the class distribution of its training rows,
  and predicts the majority class (or returns the proportions as probabilities).
- For regression, the leaf stores the mean target of its training rows.

Because every question depends on a single feature and a threshold, the regions a
tree carves out are **axis-aligned boxes**. The boundary between two classes is a
staircase of horizontal and vertical segments, never a diagonal.

## 3. When Decision Trees Fit Well

Trees are a natural choice when:

- **The data are tabular.** Rows and columns, mixed types, missing values. Trees
  handle this better than most models.
- **The relationships are nonlinear or full of interactions.** "Discount matters
  only for new customers in the north" is a rule a tree can express without any
  feature engineering.
- **Explainability matters.** A doctor, loan officer, or auditor can follow a path
  of questions. That is often more convincing than a list of coefficients.
- **You need a quick, strong baseline.** A single tree trains in seconds and tells
  you which features matter.
- **Features have different scales or units.** A split at `area_sqft <= 1800`
  does not care whether another feature is measured in metres or millimetres.

Trees are a weaker choice when:

- **The true relationship is smooth and linear.** A deep tree approximates a line
  with many steps and does it worse than a line.
- **You must extrapolate.** A tree cannot predict a price above the largest price
  it saw during training.
- **The data are very high-dimensional and sparse**, such as raw text or pixels.
  Other methods usually win there.
- **Stability is critical.** A tiny change in the training set can produce a very
  different tree.

## 4. How a Tree Makes a Prediction

Prediction is mechanical. Start at the root and follow the questions:

```text
row = {petal_length: 6.3, petal_width: 1.8}

root:  petal_length <= 2.45?   -> no
node:  petal_width  <= 1.55?   -> no
node:  petal_width  <= 1.70?   -> no
leaf:  class = virginica
```

The list of nodes visited is called the **decision path**. It is the explanation
for a single prediction: these are the exact tests the row passed, in order. If a
customer asks "why was I rejected?", the answer is the path, not a coefficient.

Two properties follow directly:

- **The path is short.** A balanced tree of depth `d` visits at most `d + 1` nodes.
  This is why trees are fast at prediction time.
- **Only some features appear.** Features that never win a split on the path are
  irrelevant to this prediction, even if they matter elsewhere in the tree.

## 5. Choosing a Split: Impurity

A tree grows by asking: **which question separates the classes best?** "Best" needs
a number. The standard number is **impurity**, a measure of how mixed the labels in
a group are. A pure group contains one class; a mixed group contains several.

For a group with class proportions `p_1, p_2, ..., p_k`, two common impurities are:

**Entropy** (from information theory):

```text
H = - sum_i p_i * log2(p_i)
```

With the convention `0 * log2(0) = 0`. Entropy is 0 for a pure group and reaches
its maximum `log2(k)` when all classes are equally likely. For two balanced
classes, `H = 1`.

**Gini impurity**:

```text
G = 1 - sum_i p_i^2
```

Gini is also 0 for a pure group and is largest when classes are balanced. For two
balanced classes, `G = 0.5`. It is slightly cheaper to compute than entropy and is
scikit-learn's default, which is why it shows up so often.

A third option, **misclassification error** `1 - max_i p_i`, is rarely used for
growing trees. It is flat in many situations and gives the greedy search little to
work with, even though it is what we ultimately care about at the leaf.

### Comparing entropy and Gini on the same group

Take a group with 80% one class and 20% the other:

| Measure | Calculation | Value |
|---------|-------------|-------|
| Entropy | `-0.8 log2 0.8 - 0.2 log2 0.2` | 0.722 |
| Gini | `1 - 0.8^2 - 0.2^2` | 0.320 |
| Misclassification error | `1 - 0.8` | 0.200 |

All three agree that this group is less mixed than a 50/50 group, but they are on
different scales. Do not compare an entropy value with a Gini value.

## 6. Information Gain

Impurity tells us how mixed one group is. To judge a **split**, compare the
impurity before and after. The parent node has impurity `I(parent)`. After the
split, each child `c` has impurity `I(c)` and holds a fraction `n_c / n` of the
rows. The weighted child impurity is:

```text
I(children) = sum_c (n_c / n) * I(c)
```

The improvement is the **information gain**:

```text
gain = I(parent) - I(children)
```

The greedy algorithm tries every feature and every candidate threshold, computes
the gain, and keeps the split with the largest gain. It does not look ahead. This
local choice is why trees are fast and why they are not guaranteed to find the
globally smallest tree.

### A split by hand

Suppose ten pieces of fruit have been labelled **sweet** or **not sweet**. The
whole group has five of each, so its entropy is exactly 1 bit. Consider two
candidate questions.

**Split A — colour:** five red and five green.

- Red: 4 sweet, 1 not → entropy 0.722
- Green: 1 sweet, 4 not → entropy 0.722
- Weighted children: `0.5 * 0.722 + 0.5 * 0.722 = 0.722`
- Gain: `1.000 - 0.722 = 0.278`

**Split B — size:** five big and five small.

- Big: 3 sweet, 2 not → entropy 0.971
- Small: 2 sweet, 3 not → entropy 0.971
- Weighted children: `0.5 * 0.971 + 0.5 * 0.971 = 0.971`
- Gain: `1.000 - 0.971 = 0.029`

Colour separates the labels far better, so the tree splits on colour first. Notice
that both splits are "balanced" in the sense of having equal-sized children. The
difference is **how pure** each child becomes, not how many rows it holds.

### Why not simply maximise accuracy?

Accuracy is measured after the leaf chooses a class. A split that isolates two rows
of the same class and leaves everything else mixed can look like a big accuracy
win while doing very little. Impurity rewards moving the whole distribution toward
purity, which produces more useful probabilities and better subsequent splits.

## 7. The Basic Learning Algorithm

Tree induction is **recursive partitioning**. In words:

```text
grow(rows):
    if stop(rows):
        return a leaf predicting the majority class or mean target
    find the feature and threshold with the highest gain
    if no split improves impurity:
        return a leaf
    split rows into left and right
    return a node(question, grow(left), grow(right))
```

In more detail:

1. Start with all training rows at the root.
2. For each feature, sort the values and consider thresholds between adjacent
   distinct values.
3. Score every candidate split with information gain (or Gini reduction).
4. Keep the best split, partition the rows, and create two children.
5. Repeat on each child until a stopping rule fires.
6. At each leaf, store the majority class or the mean target.

The algorithm is **greedy**: each split is the best one available now, with no
promise that it is part of the best tree overall. Finding the globally optimal
tree is computationally infeasible for realistic data, so every practical tree
learner is greedy.

The algorithm is also **recursive**, which is why the natural code shape is a
function that calls itself on the left and right subsets.

## 8. Stopping Rules

If we keep splitting until every leaf is pure, the tree will memorize the training
set. On noisy data that means fitting individual mistakes. We need rules that stop
the growth, or that remove branches afterwards.

Common **pre-pruning** controls, applied during growth:

| Parameter | Stops when |
|-----------|------------|
| `max_depth` | the path from root reaches a given length |
| `min_samples_split` | a node has fewer rows than the minimum needed to split |
| `min_samples_leaf` | a split would create a child with too few rows |
| `max_leaf_nodes` | the tree has reached a maximum number of leaves |
| `min_impurity_decrease` | the best split improves impurity by less than a threshold |

Pre-pruning is cheap and easy to explain, but it has a flaw: it can stop too early.
A split that looks weak now may enable a very strong split one level below. Because
the algorithm is greedy, it cannot know that.

**Post-pruning** grows a full tree and then removes branches. The most common form
in scikit-learn is **cost-complexity pruning**, controlled by `ccp_alpha`:

```text
R_alpha(T) = R(T) + alpha * |T|
```

Here `R(T)` is the training error of tree `T` and `|T|` is its number of leaves.
Larger `alpha` penalizes large trees more, so the tree is pruned back. We choose
`alpha` with cross-validation, not by looking at the test set.

## 9. Overfitting in Trees

An unconstrained tree has three linked symptoms:

- **Training accuracy near perfect**, often exactly 1.0.
- **Test accuracy much lower** and erratic.
- **Many leaves**, some containing only one or two rows.

The tree has turned noise into rules. In a house-price tree, for example, it may
carve out a leaf for "area = 2,171 and age = 11 and location = Town" because one
such house exists. That rule will not generalize.

The cure is the same as in earlier days: constrain capacity and validate honestly.
The practical will show training and test scores side by side as `max_depth`
changes. The pattern to look for is the classic U-shape: test performance improves
as the tree grows, peaks at some moderate depth, then degrades while training
performance keeps climbing.

Two useful rules of thumb:

- **A single tree rarely needs to be deep.** Depths of 3 to 6 capture most of the
  signal in ordinary tabular problems.
- **Look for the smallest tree that is close to the best validation score.** The
  extra branches are usually fitting noise.

## 10. Pruning by Hand: A Contrast

Consider a branch that splits off a single training row and predicts it correctly:

```text
... -> [371 rows: 200 yes, 171 no]
         |
      feature_x <= 4.2
        /          \
   [370 rows]    [1 row: no]
   predict yes   predict no
```

The leaf on the right improves training accuracy by one row. It also adds a rule
that exists for exactly one observation. Removing that branch changes almost
nothing on training data and usually improves test behaviour. That is the intuition
behind both `min_samples_leaf` and cost-complexity pruning.

## 11. Trees for Regression

Everything above applies to regression with one substitution: instead of class
proportions, measure how spread out the target is. The usual criterion is the
**squared error** within a node:

```text
MSE(node) = (1/n) * sum_i (y_i - y_bar)^2
```

where `y_bar` is the mean target in the node. A split is good if the weighted MSE
of the children is much lower than the MSE of the parent. At a leaf, the prediction
is `y_bar`.

The result is a **piecewise-constant** function: a staircase that approximates the
data. Deeper trees add more steps. Once a leaf contains one row, its MSE is zero,
which is exactly the overfitting we saw in classification.

Regression trees share the classification tree's inability to extrapolate. A
house-price tree trained on prices up to 300,000 cannot predict 500,000, because no
leaf has seen it. A linear model, for better or worse, will happily continue the
trend.

## 12. Feature Importance

Trees offer a free ranking of features. Every split reduces impurity by some
amount, weighted by the number of rows that pass through the node. Summing those
reductions per feature and normalizing gives **impurity-based feature importance**,
available as `model.feature_importances_`.

It is useful but has two known biases:

- **It favours features with many distinct values**, because they offer more
  thresholds to split on. A unique identifier can appear important while being
  useless.
- **It measures usefulness inside this tree**, not causal effect. Two correlated
  features may split the credit unpredictably.

A more reliable alternative is **permutation importance**: shuffle one feature in
the validation set and measure how much performance drops. If the drop is small,
the model was not relying on that feature. It is slower but less biased.

## 13. Instability and Axis-Aligned Boundaries

Trees have two structural quirks worth remembering.

**Instability.** Change a few training rows and the best root split can change.
Once the root changes, every subtree below it changes too, so two nearly identical
datasets can produce very different trees. This is why a single tree's feature
importances can look arbitrary. It is also the motivation for **ensembles** such
as random forests and gradient boosting, which average many trees to reduce this
variance. Those are the subject of Day 11.

**Axis-aligned boundaries.** Because every split tests one feature at a time, the
boundary between classes is a staircase. A diagonal boundary must be approximated
by many small steps. If the true boundary is a smooth diagonal, a tree needs more
depth than a linear model needs parameters, and it may still look jagged.

```text
diagonal truth        tree approximation
      /                     _|
     /                    _|
    /                   _|
   /                  _|
```

## 14. Strengths and Limitations

**Strengths**

- Easy to interpret and visualize.
- No feature scaling required.
- Handles nonlinearity and interactions automatically.
- Works with mixed feature types after encoding.
- Fast to train and fast to predict.
- Provides feature importance and per-row explanations.
- A natural building block for ensembles.

**Limitations**

- Prone to overfitting without pruning.
- Unstable to small changes in the data.
- Axis-aligned boundaries are a poor fit for smooth diagonal patterns.
- Cannot extrapolate beyond the training range.
- Impurity-based importance is biased.
- A single tree is usually outperformed by an ensemble.

## 15. Real-World Use Case: Loan Default Screening

A lender wants a model that flags applications likely to default. Regulators require
that a rejection be explainable, so a black box is unattractive. A shallow decision
tree is a good first model.

- **Features:** income, loan amount, credit score, employment length, existing debt.
- **Target:** default within 12 months, yes or no.
- **Tree:** depth 4, `min_samples_leaf=50`, Gini impurity.
- **Explanation:** a rejected applicant receives the exact path, for example
  "debt-to-income above 0.42 and credit score below 610".
- **Validation:** train on three years of history, test on a later year to respect
  the time order.
- **Guardrail:** check that protected attributes are excluded, and that no single
  leaf is so small that it encodes an individual.

The tree will not be the final production model; a gradient-boosted ensemble or a
regularized logistic model often scores better. But the tree sets an interpretable
baseline and exposes the rules the data actually supports.

## 16. Common Mistakes

- **Leaving the tree unconstrained.** A perfect training score is a warning, not a
  victory.
- **Tuning `max_depth` on the test set.** Use cross-validation; touch the test set
  once at the end.
- **Comparing impurity values across criteria.** Entropy and Gini live on different
  scales.
- **Encoding categories poorly.** Ordinal integers can imply a false order;
  one-hot encoding is safer for nominal categories.
- **Trusting feature importance blindly.** It is biased toward high-cardinality
  features and says nothing about causation.
- **Using a single tree where an ensemble is affordable.** If interpretability is
  not the hard requirement, Random Forest or boosting is usually stronger.
- **Expecting extrapolation.** A tree cannot predict outside the range of its
  training targets.
- **Forgetting the baseline.** A tree at 85% accuracy may be no better than always
  predicting the majority class.

## 17. Putting It Together

The whole method fits in a few lines:

1. Represent the data as rows of features and a target.
2. Define impurity: entropy or Gini for classification, MSE for regression.
3. Grow the tree greedily, always choosing the split with the largest impurity
   reduction.
4. Stop with depth, leaf-size, or gain constraints.
5. Prune with `ccp_alpha` if pre-pruning was too blunt.
6. Evaluate on held-out data against a baseline.
7. Read the tree, the decision paths, and the feature importances to explain it.

Every later tree-based method, from random forests to gradient boosting, repeats
steps 3 and 4 many times with variation. Understanding one tree well makes the
ensembles much easier to understand.

## 18. Summary

- A decision tree predicts by routing a row through yes/no questions to a leaf.
- The root, internal nodes, branches, and leaves form a drawable flowchart.
- Splits are chosen greedily by impurity reduction: information gain for entropy
  or Gini reduction for Gini.
- Entropy and Gini are both 0 for pure nodes and largest for balanced ones.
- Unconstrained trees overfit; `max_depth`, `min_samples_leaf`, and `ccp_alpha`
  control capacity.
- Trees handle nonlinearity and interactions but are unstable, cannot extrapolate,
  and produce axis-aligned boundaries.
- Feature importance is a useful but biased summary.
- A single tree is an interpretable baseline; ensembles build on the same ideas.

## Outcome

Learners can describe a decision tree in terms of nodes, branches, and leaves,
compute entropy, Gini impurity, and information gain for a candidate split, explain
the greedy recursive partitioning algorithm, choose stopping and pruning settings
to control overfitting, interpret a decision path and feature importances, and
state the strengths and limitations of tree models.
