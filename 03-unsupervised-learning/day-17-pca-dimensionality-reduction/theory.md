# Day 17 — Theory: Dimensionality Reduction and PCA

**Module:** Module 4: Unsupervised Learning

More features are not always better. Every extra feature adds noise, cost, and a
chance to overfit. It also spreads the data thinner: in high dimensions, points that
look nearby may be far apart, and distances lose meaning. **Dimensionality
reduction** attacks this problem by replacing many features with a smaller set that
keeps most of the information.

The most common method is **principal component analysis (PCA)**. It finds new axes
that point in the directions of greatest variation, then projects the data onto a
few of them. The result is a compressed representation that is often good enough for
visualization, denoising, or a downstream model, and sometimes faster and more
accurate than the original.

## In Plain English

**Think of it like this:** PCA finds the few directions in which the data
stretches the most, then describes every point using only those directions.

- It **rotates the axes** so the first new axis follows the biggest spread, the
  second the next biggest, and so on.
- Keep only the first few components to compress the data.
- **Standardize features first**, because PCA is based on variance.
- Use a **scree plot** or cumulative explained variance to choose how many
  components to keep.
- PCA compresses and denoises, but the new axes are harder to interpret.

## 1. The Curse of Dimensionality

Adding dimensions has three related costs:

- **Sparsity.** A fixed number of points covers a smaller fraction of the space as
  dimensions grow. To keep the same density, the sample size must grow
  exponentially.
- **Distance concentration.** The distance to the nearest and farthest neighbors
  become similar in high dimensions, so "nearest" stops being informative.
- **Combinatorial models.** The number of possible feature combinations explodes,
  making it hard to learn interactions without enormous data.

High-dimensional data also make visualization impossible and computation slower. Yet
many real datasets have far more columns than rows: text has one column per word,
images have one per pixel, and genomics has thousands of genes. Dimensionality
reduction is not a luxury in those settings; it is a necessity.

## 2. Feature Selection versus Feature Extraction

There are two broad approaches to fewer features:

**Feature selection** keeps a subset of the original features and discards the
rest. The result is still interpretable in terms of the original columns.

- Filter methods rank features by a statistic (correlation, mutual information).
- Wrapper methods search subsets using a model's performance.
- Embedded methods select during training (Lasso, tree importance).

**Feature extraction** creates new features as combinations of the old ones. PCA is
the canonical example. The new features are usually not interpretable as any single
original column, but they can capture structure that no individual column does.

| Aspect | Selection | Extraction |
|--------|-----------|------------|
| Output | subset of original columns | new combined columns |
| Interpretability | high | lower |
| Information use | discards some features | mixes all features |
| Example | drop low-variance columns | PCA |

Choose selection when you need to explain the model in terms of original
measurements. Choose extraction when compression and downstream performance matter
more than interpretability.

## 3. What PCA Does

PCA looks for a sequence of new axes, called **principal components**, with three
properties:

1. They are **orthogonal** (uncorrelated) to each other.
2. The first component points in the direction of **maximum variance**.
3. Each later component captures the maximum remaining variance, subject to being
   orthogonal to the earlier ones.

Projecting the data onto the first `k` components gives a `k`-dimensional
representation. If the data lie close to a lower-dimensional subspace, a small `k`
retains most of the variation.

A geometric picture for two features:

```text
       .
     .   .
    .  .  .          PC1 runs along the long axis
   .  .  .           of the cloud.
  .  .  .
```

The long axis of the ellipse is where the data vary most. PCA rotates the coordinate
system so that this axis becomes the first coordinate. The second component is
perpendicular and captures the thinner direction.

## 4. The Steps

PCA is a sequence of linear-algebra operations:

1. **Center** each feature by subtracting its mean.
2. **Scale** each feature to unit variance (usually).
3. Compute the **covariance matrix** of the centered data.
4. Compute its **eigenvalues** and **eigenvectors**.
5. Sort eigenvectors by decreasing eigenvalue.
6. Project the data onto the top `k` eigenvectors.

The eigenvectors are the principal components; the eigenvalues measure the variance
captured by each.

### Why eigenvectors

The covariance matrix summarizes how features move together. Its eigenvectors are
the directions that the matrix only stretches, not rotates. The amount of stretch is
the eigenvalue. The direction with the largest eigenvalue is therefore the direction
of largest variance, which is exactly what PCA wants.

### A two-feature example

Suppose two standardized features have correlation 0.9. Their covariance matrix is:

```text
[ 1.0  0.9 ]
[ 0.9  1.0 ]
```

Its eigenvectors are `(1, 1)` and `(1, -1)`, with eigenvalues about 1.9 and 0.1. The
first component is the sum of the features, the second their difference. The first
captures almost all the variance because the features move together. Proceeding with
one component loses little information.

## 5. Standardization

PCA is variance-based, so the scale of each feature matters. A feature measured in
millimetres has 1,000 times the variance of the same feature measured in metres. PCA
will treat the millimetre version as far more important, which is rarely intended.

**Always standardize** features before PCA unless they are already on a common,
meaningful scale. Standardization gives each feature a chance to contribute.

Caveat: standardizing is not always right. If the units are meaningful and larger
variance really does mean more importance, you may skip it. The decision should be
explicit, not accidental.

## 6. Explained Variance

Each principal component has an associated **explained variance ratio**: the
fraction of total variance it captures.

```text
explained variance ratio_i = eigenvalue_i / sum of all eigenvalues
```

The ratios sum to 1 across all components. Plotting them in order gives a **scree
plot**, and plotting their cumulative sum shows how many components are needed to
retain a target fraction, such as 90% or 95%.

```text
variance explained
  1.0 |            ___________
      |        ___/
      |    ___/
      |___/
      +----------------------- components
         2  4  6  8 10
```

Choosing `k` is a trade-off. More components retain more information but reduce the
benefit of compression. Common choices:

- **95% cumulative variance** as a default for downstream models.
- **Enough components for visualization**, usually the first two.
- **A scree-plot elbow** where added components stop contributing much.

Explained variance is not the same as predictive usefulness. A component may capture
variance that is pure noise for the target, and a low-variance component may be
highly predictive. Always validate the downstream task.

## 7. Reconstruction and Information Loss

Because PCA is a linear projection, we can map the reduced data back to the original
feature space:

```text
X_original_approx = Z_reduced @ components^T + mean
```

The difference between the original and the reconstruction is the **reconstruction
error**. With all components, the error is zero. With fewer, it grows. This error is
a concrete measure of what was lost.

```text
error = mean( (X - X_approx)^2 )
```

Low reconstruction error with few components means the data really do lie near a
low-dimensional subspace. High error means PCA is throwing away structure that may
matter.

## 8. PCA for Visualization

For tabular data with many features, plotting the first two components is a quick way
to see structure: clusters, gradients, or outliers. Iris is the classic example. Four
measurements become two coordinates that still separate the species reasonably well.

But a 2D projection can distort. Points that look close may be far apart in the full
space, and clusters that look separate may overlap. Use the plot to form hypotheses,
not to prove them.

## 9. PCA Before a Supervised Model

PCA is often used as a preprocessing step:

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.95),
    LogisticRegression(max_iter=1000),
)
```

Benefits:

- Fewer features means faster training and prediction.
- Noise in low-variance directions is removed.
- Correlated features are collapsed, which can stabilize some models.

Risks:

- Useful low-variance signal may be discarded.
- Components are harder to interpret.
- The preprocessing must be fit inside the pipeline, or leakage occurs.

**Leakage warning:** fitting PCA on the full dataset before splitting lets information
from the test set shape the components. Always fit the scaler and PCA on the training
fold only. A pipeline handles this automatically during cross-validation.

## 10. Limitations of PCA

- **It is linear.** It finds linear projections. Curved manifolds need nonlinear
  methods such as t-SNE, UMAP, or autoencoders.
- **It is variance-driven.** High variance is not the same as importance. A feature
  can vary a lot and be irrelevant; a small but consistent signal can be predictive.
- **It assumes directions are orthogonal.** Real structure may be oblique.
- **It is sensitive to scaling.** The result changes completely if features are not
  standardized.
- **It is sensitive to outliers.** A few extreme points can dominate the components.
- **It is less interpretable.** Components are mixtures of all features.
- **It can hide class structure.** The top variance directions may not separate the
  classes.

## 11. Applications

- **Visualization:** project high-dimensional data to two dimensions.
- **Compression:** store fewer numbers per observation.
- **Denoising:** discard low-variance components that are mostly noise.
- **Preprocessing:** speed up models and reduce multicollinearity.
- **Anomaly detection:** points with high reconstruction error are unusual.
- **Image processing:** the classic eigenfaces representation of faces.

PCA is also used as a first step before clustering or nearest-neighbor search, where
high dimensions are especially harmful.

## 12. Common Mistakes

- **Forgetting to standardize.** The first component becomes the largest-unit
  feature.
- **Choosing `k` by explained variance alone.** Validate the actual task.
- **Fitting PCA on all data before splitting.** This leaks.
- **Assuming PCA always improves a model.** It can remove useful signal.
- **Interpreting components as original features.** They are mixtures.
- **Using PCA for nonlinear structure.** Try a manifold method instead.
- **Ignoring outliers.** They can define the first component.
- **Believing two components tell the whole story.** Plots distort.

## 13. Summary

- High dimensions bring sparsity, distance concentration, and cost.
- Feature selection keeps columns; feature extraction creates combinations.
- PCA finds orthogonal directions of maximum variance.
- It is computed from the eigenvectors of the covariance matrix.
- Standardize features before PCA.
- Explained variance ratio and scree plots guide the choice of `k`.
- Reconstruction error measures the information lost.
- PCA can speed up and denoise downstream models but can also discard signal.
- Components are less interpretable, and PCA is linear and variance-driven.

## Outcome

Learners can explain the curse of dimensionality, distinguish selection from
extraction, describe how PCA finds principal components, standardize data correctly,
use explained variance and scree plots to choose `k`, measure reconstruction error,
apply PCA in a leak-free pipeline, and state its limitations.
