# Day 10 — Support Vector Machines (SVM)

**Module:** Module 2: Supervised Learning

Logistic regression finds *a* line that separates the classes. A **support vector
machine** finds the line with the widest empty corridor around it. That geometric
goal, called the **maximum margin**, makes SVMs robust and gives them a clean
theoretical story: the boundary depends only on the few points closest to it, the
**support vectors**.

The second half of the story is the **kernel trick**. By replacing dot products
with similarity functions, an SVM can draw curved boundaries without explicitly
building new features. Linear and RBF kernels turn one algorithm into a family that
handles text, images, and small nonlinear datasets.

## What You Will Learn

- What a separating hyperplane is, and how many can fit the same data.
- Why the maximum-margin hyperplane is preferred.
- How the margin width relates to `||w||`.
- What support vectors are, and why other points do not affect the boundary.
- How the soft-margin parameter `C` trades margin width for training errors.
- How the dual form depends only on dot products.
- What the kernel trick does, and the difference between linear, polynomial, and
  RBF kernels.
- How `gamma` controls the flexibility of an RBF boundary.
- Why feature scaling is mandatory for SVMs.
- When a linear SVM beats a nonlinear one, and how to tune `C` and `gamma`.

## Detailed Content

- [Theory](theory.md) — hyperplanes, maximum margins, support vectors, soft
  margins and `C`, the dual form, the kernel trick, RBF and `gamma`, SVR,
  scaling, multiclass SVMs, strengths, limitations, and common mistakes.
- [Practical](practical.md) — compute a distance to a hyperplane, train and plot a
  linear SVM, count support vectors, sweep `C`, compare kernels, study `gamma`,
  demonstrate a scaling failure, benchmark against logistic regression, and tune
  an RBF SVM on breast cancer data.

## Files

No CSV is needed. Every dataset is generated or bundled with scikit-learn:

- **Blobs** (`make_blobs`) — a clean, linearly separable binary problem.
- **Two moons** (`make_moons`) — a curved binary problem for comparing kernels.
- **Breast Cancer Wisconsin** (`load_breast_cancer`) — 569 rows, 30 features, two
  classes, for a realistic benchmark.

The notebook should be created inside this Day 10 folder. Nothing is downloaded.

## Before You Start

- Day 07: classification, logistic regression, and decision boundaries.
- Day 06: feature scaling and regularization.
- Day 04: train/test splits, baselines, and honest evaluation.

## Outcome

Learners can explain maximum-margin classification, identify support vectors,
describe how `C` and slack control overfitting, explain the kernel trick and the
role of `gamma`, build a leak-free scaling pipeline, tune an SVM with
cross-validation, and choose between linear and nonlinear SVMs by comparing honest
held-out results.
