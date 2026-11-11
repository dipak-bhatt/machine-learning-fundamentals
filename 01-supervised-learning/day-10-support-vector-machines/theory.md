# Day 10 — Theory: Support Vector Machines

**Module:** Module 2: Supervised Learning

Logistic regression separates classes by fitting a probability model. When many
lines separate the data equally well, it is happy with whichever one the optimizer
lands on. A **support vector machine (SVM)** asks a sharper question: among all
separating lines, which one sits as far as possible from the nearest points of
either class? That is the **maximum-margin** idea, and it gives the model a
geometric backbone that is easy to reason about and surprisingly powerful.

The second idea is the **kernel trick**. It lets a linear method draw curved
boundaries without ever computing the curved coordinates explicitly. Together,
margins and kernels made SVMs the dominant classifier for a decade before deep
learning, and they remain an excellent choice for small, high-dimensional, and
clean datasets.

## In Plain English

**Think of it like this:** an SVM draws the **widest possible street** between
two classes and keeps new points as far from the curb as it can.

- The **margin** is the empty gap around the boundary; the nearest points are
  the **support vectors**.
- Maximizing the margin is the same as minimizing the length of `w`.
- **`C`** controls strictness: large `C` = strict and narrow, small `C` = wide
  and tolerant.
- **Kernels** let a linear method draw curved boundaries without computing the
  extra coordinates.
- SVMs shine on small, clean, high-dimensional data; always scale features
  first.

## 1. From a Line to a Margin

A binary classifier must place a boundary between the classes. For two features
that boundary is a line; in higher dimensions it is a **hyperplane**. Many
hyperplanes can separate the same training data:

```text
        o o o                     o o o
      o o o o                   o o o o
   ---------------           ----------------
     x x x x                   x x x x
       x x                       x x
```

The left boundary hugs the negative class; the right one hugs the positive class.
Both achieve zero training error. If a new point arrives near the boundary, which
one would you trust? The intuitive answer is: the one in the middle, with the
largest empty corridor on each side. The SVM finds exactly that corridor.

**Margin** is the distance from the boundary to the closest training point. The
maximum-margin classifier chooses the hyperplane that maximizes this distance. The
corridor is called the **margin band**, and the points that touch its edges are the
**support vectors**.

## 2. The Hyperplane and the Score

A hyperplane in `p` dimensions is the set of points satisfying:

```text
w . x + b = 0
```

Here `w` is the normal vector (it points perpendicular to the hyperplane), `b` is
the bias, and `w . x` is the dot product. The signed score `w . x + b` tells us
which side of the boundary a point is on:

- `w . x + b > 0` → predict the positive class
- `w . x + b < 0` → predict the negative class
- `w . x + b = 0` → the point lies exactly on the boundary

The distance from a point to the hyperplane is:

```text
distance = |w . x + b| / ||w||
```

where `||w||` is the length of `w`. This formula is the key to the margin. If we
constrain the closest points to have `|w . x + b| = 1`, their distance becomes
`1 / ||w||`. The margin band has width `2 / ||w||`, so **maximizing the margin is
the same as minimizing `||w||`**.

## 3. The Maximum-Margin Problem

Encode the labels as `y_i = +1` or `-1`. A hyperplane separates the data correctly
with a margin if:

```text
y_i * (w . x_i + b) >= 1   for every training point i
```

The left side is the signed score flipped by the label, so it is positive when the
point is on the correct side and at least 1 when it is outside the margin band.

The optimization problem is then:

```text
minimize     (1/2) ||w||^2
subject to   y_i * (w . x_i + b) >= 1
```

The square and the half make the derivative clean. This is a **convex quadratic
program**: there is a single global optimum, unlike a decision tree's greedy
search. That is one of the SVM's attractive properties.

### Why only the support vectors matter

The constraints that are active, meaning they hold with equality, are the points
exactly on the margin band. All other points satisfy the inequality strictly and
could move slightly without changing the solution. Therefore the optimal
hyperplane depends only on the support vectors. Remove any other training point and
the boundary does not move.

This is a striking difference from logistic regression, where every point
contributes to the loss. It also means SVMs can be memory-efficient at prediction
time: only the support vectors need to be stored.

## 4. Soft Margins and the Parameter C

Real data are rarely perfectly separable. A single mislabelled point could make the
hard-margin problem infeasible. The **soft-margin** SVM introduces a slack variable
`xi_i >= 0` for each point and allows controlled violations:

```text
minimize     (1/2) ||w||^2 + C * sum_i xi_i
subject to   y_i * (w . x_i + b) >= 1 - xi_i
             xi_i >= 0
```

The trade-off is explicit:

- **Large `C`** punishes violations heavily. The model tries to classify every
  training point correctly and uses a narrow margin. It can overfit.
- **Small `C`** tolerates violations. The model prefers a wide, smooth margin even
  if some points are inside it or on the wrong side. It regularizes more.

`C` plays a role similar to the inverse of the regularization strength in Ridge and
Lasso. It is a hyperparameter chosen with cross-validation, not learned from the
data.

| `C` | Margin | Training errors | Risk |
|-----|--------|-----------------|------|
| Very small | wide | many allowed | underfitting |
| Moderate | balanced | few | usually best |
| Very large | narrow | almost none | overfitting |

## 5. The Dual Form and Why It Matters

The constrained problem has a **dual** form, derived with Lagrange multipliers
`alpha_i >= 0`, one per training point:

```text
maximize     sum_i alpha_i - (1/2) sum_i sum_j alpha_i alpha_j y_i y_j (x_i . x_j)
subject to   sum_i alpha_i y_i = 0
             0 <= alpha_i <= C
```

Two facts make this form important:

- Most `alpha_i` are zero. The points with `alpha_i > 0` are the support vectors.
- The data appear **only through dot products** `x_i . x_j`.

The second fact is the doorway to kernels. If we can replace every dot product
with a function `K(x_i, x_j)` that measures similarity, we can work in a richer
space without ever building its coordinates.

## 6. The Kernel Trick

A **kernel function** computes the dot product of two points after they have been
mapped into a higher-dimensional space:

```text
K(x, z) = phi(x) . phi(z)
```

The trick is that we never compute `phi(x)` explicitly. We only need `K`, and the
dual optimization uses `K` in place of the dot product. This turns a linear
classifier into a nonlinear one at the cost of evaluating `K` for pairs of points.

Common kernels:

| Kernel | Formula | Behavior |
|--------|---------|----------|
| Linear | `x . z` | a straight hyperplane; good for text and high-dimensional sparse data |
| Polynomial | `(gamma * x . z + r)^d` | curved boundaries; `d` controls the degree |
| RBF (Gaussian) | `exp(-gamma * ||x - z||^2)` | local, flexible boundaries; a strong default |
| Sigmoid | `tanh(gamma * x . z + r)` | resembles a neural network; less common |

### The RBF kernel

The **radial basis function** kernel measures similarity by distance:

```text
K(x, z) = exp(-gamma * ||x - z||^2)
```

- If `x` and `z` are close, the exponent is near 0 and `K` is near 1.
- If they are far apart, `K` approaches 0.

The parameter `gamma` controls how quickly similarity fades with distance:

- **Large `gamma`**: each point influences only a small neighbourhood. The boundary
  becomes wiggly and can overfit.
- **Small `gamma`**: influence reaches far. The boundary becomes smooth and can
  underfit.

`gamma` and `C` are the two knobs to tune for an RBF SVM. A useful intuition is
that `C` controls how much the model cares about individual mistakes, while
`gamma` controls how far each training point's influence extends.

### What the kernel cannot do

The kernel trick is not magic. It does not add information; it changes the geometry
in which the model looks for a linear separator. If the problem needs a feature
that is not a function of the inputs, no kernel creates it. It also makes
prediction more expensive, because the model must compare a new point with many
support vectors.

## 7. SVM for Regression

The same margin idea extends to regression through **support vector regression
(SVR)**. Instead of a margin band between classes, SVR fits a tube of width
`epsilon` around the predictions and penalizes points that fall outside it. Inside
the tube, errors are ignored. Outside, they are penalized linearly, controlled by
`C`.

SVR is useful when you want a robust, margin-based regressor, but it is less common
than tree-based regressors in practice. The classification ideas are the core of
this day.

## 8. Feature Scaling Is Not Optional

The margin is measured with `||w||`, and distances combine all features. If one
feature ranges over thousands and another over fractions, the large one dominates
the geometry. The SVM will effectively ignore the small one.

Always standardize features before an SVM, especially with an RBF kernel. The
practical will show a case where scaling changes the result dramatically.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
```

Wrapping the scaler in a pipeline also prevents leakage: the scaler is fit on the
training fold only.

## 9. Multiclass Classification

SVMs are fundamentally binary. scikit-learn's `SVC` handles multiclass problems
with **one-vs-one (OvO)**: it trains a binary classifier for every pair of classes
and lets them vote. For `k` classes that is `k(k-1)/2` classifiers, which is
manageable for small `k` but grows quickly. `LinearSVC` uses one-vs-rest instead.

## 10. Strengths and Limitations

**Strengths**

- Effective in high-dimensional spaces, including when `p` is larger than `n`.
- Convex objective: a single, reproducible global optimum.
- Memory-efficient at prediction time because only support vectors matter.
- Kernels give flexible nonlinear boundaries without manual feature engineering.
- Strong theoretical foundation and good performance on clean, medium-sized data.

**Limitations**

- Training time grows quickly with the number of rows; standard solvers are not
  ideal for millions of examples.
- Requires careful scaling.
- Two hyperparameters (`C` and `gamma`) interact and need tuning.
- Probabilities are not native; `probability=True` uses an expensive calibration.
- Hard to interpret: the model is a weighted combination of many support vectors.
- Multiclass training multiplies the number of binary problems.

## 11. Real-World Use Case: Text Classification

SVMs earned much of their reputation on text. A spam filter or topic classifier
often has tens of thousands of features (one per word or n-gram) and a few thousand
rows. Linear SVMs handle this shape well:

- Features are sparse: most words do not appear in a given document.
- The classes are often close to linearly separable in TF-IDF space.
- The linear kernel is fast and interpretable through its weights.
- Only a subset of documents becomes support vectors.

A typical pipeline is TF-IDF followed by `LinearSVC`. It remains a strong baseline
for spam detection, sentiment analysis, and document routing, even in the era of
transformers.

## 12. Common Mistakes

- **Forgetting to scale.** The most common SVM bug by far.
- **Tuning `gamma` without `C` (or vice versa).** They interact; tune them
  together with a grid or randomized search.
- **Using `probability=True` casually.** It is much slower; use decision scores for
  ranking when possible.
- **Assuming the RBF kernel is always better.** On text or very high-dimensional
  data, a linear kernel often wins.
- **Reporting training accuracy.** SVMs can fit training data almost perfectly with
  a large `C`; only held-out scores tell you anything.
- **Ignoring class imbalance.** `class_weight="balanced"` is often necessary.
- **Trying to interpret `w` for a nonlinear kernel.** The coefficients live in an
  implicit space and do not map cleanly to input features.

## 13. Putting It Together

The SVM story in five steps:

1. A hyperplane separates classes; its distance to the nearest points is the
   margin.
2. Maximizing the margin means minimizing `||w||` subject to correct-side
   constraints.
3. The soft-margin version adds slack and the penalty `C` to handle noisy data.
4. The dual form depends only on dot products, enabling kernels.
5. A kernel such as RBF replaces the dot product with a similarity, producing
   nonlinear boundaries.

Scale the features, choose a kernel, tune `C` and `gamma` with cross-validation,
and compare against a simple baseline.

## 14. Summary

- SVMs find the maximum-margin separating hyperplane.
- Support vectors are the points on the margin; they alone determine the boundary.
- `C` trades margin width against training violations.
- The kernel trick lets a linear method operate in a richer space through dot
  products.
- RBF kernels need both `C` and `gamma` tuned.
- Scaling is mandatory, training can be slow on large data, and probabilities are
  costly.
- Linear SVMs remain excellent for high-dimensional text; RBF SVMs are strong on
  small, clean, nonlinear datasets.

## Outcome

Learners can explain maximum-margin classification, identify support vectors,
describe the role of `C` and slack, explain the kernel trick and the effect of
`gamma`, apply scaling and cross-validation correctly, and choose between linear
and nonlinear SVMs for a given problem.
