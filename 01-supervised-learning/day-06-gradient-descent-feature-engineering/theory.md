# Day 06 — Theory: Gradient Descent and Feature Engineering

**Module:** Module 2: Supervised Learning

On Day 05 we learned that a straight line can predict a number, and that **least
squares** chooses the line with the smallest squared error. Then scikit-learn
produced the slope and intercept almost instantly. We never asked *how* the
library found those numbers.

Today we answer that question with **gradient descent**, a repeated calculation
that improves the parameters step by step. We then look at three ways of
changing the inputs so that a linear model can fit more interesting patterns:
**feature scaling**, **polynomial features**, and **regularization** (Ridge and
Lasso). Each idea is built from small examples before we touch real data.

## In Plain English

**Think of it like this:** gradient descent is walking downhill on an error
landscape, taking small steps in the steepest downward direction.

- Start with a guess, measure the slope of the error, and step **opposite** the
  slope.
- The **learning rate** is the step size: too big overshoots, too small crawls.
- **Scaling features** puts columns on comparable footing so training is
  smoother.
- **Polynomial features** let a line fit curves; **regularization** stops it
  from overfitting.
- **Ridge** shrinks weights; **Lasso** can shrink some weights to exactly zero.

## 1. What Day 05 Left Open

In Day 05, the model's job was:

```text
choose b0 and b1 so that the squared errors are as small as possible
```

There are two broad ways to do that.

| Approach | How it works | Strengths | Weaknesses |
|----------|--------------|-----------|------------|
| Closed-form solution | Solve a formula directly for the best coefficients | Exact, fast for a small number of features | Can be expensive with many features; needs a well-behaved matrix; only works when a formula exists |
| Iterative search | Start with a guess and improve it repeatedly | Scales to many rows and features; works for models with no formula | Takes many steps; needs tuning (learning rate) |

For linear regression, the closed-form answer is the **normal equation**:

```text
theta = (X^T X)^-1 X^T y
```

Here `theta` is the vector of coefficients, `X` is the table of features with a
column of ones added for the intercept, and `y` is the target. scikit-learn's
`LinearRegression` uses a stable version of this calculation. It does **not** use
gradient descent.

Gradient descent is the second approach. It is worth learning because:

- It works when the normal equation is too slow or impossible, for example with
  very many features.
- It is the same idea used to train logistic regression and neural networks,
  where no simple closed-form answer exists.
- It lets a model keep improving as new data arrives, one batch at a time.

Done correctly on linear regression, both approaches should find (almost) the
same line. That gives us a way to check our implementation.

## 2. The Model and Its Parameters

For one input feature, the prediction is:

```text
y_hat = b + w * x
```

- `x` is the input feature.
- `y_hat` (y-hat) is the model's prediction.
- `b` is the **intercept**: the prediction when `x = 0`.
- `w` is the **slope**, also called the **weight** or **coefficient**.

For example, if `b = 1`, `w = 2`, and `x = 3`, then:

```text
y_hat = 1 + 2 * 3 = 7
```

With several features, each feature gets its own weight:

```text
y_hat = b + w1*x1 + w2*x2 + ... + wp*xp
```

It is convenient to write this in vector form. Let `x` include a leading `1`, and
let `theta` hold the intercept and weights:

```text
x     = [1, x1, x2, ..., xp]
theta = [b, w1, w2, ..., wp]
y_hat = theta^T x
```

| Symbol | Meaning | Typical shape |
|--------|---------|---------------|
| `n` | number of training rows | a number |
| `p` | number of features | a number |
| `x_i` | the features of row `i` | one row |
| `y_i` | the true target of row `i` | one number |
| `y_hat_i` | the model's prediction for row `i` | one number |
| `theta` | all learnable parameters | `p + 1` numbers |
| `alpha` | the learning rate | a small number |

Learning means choosing `theta`. Everything that follows is a method for making
that choice.

## 3. Measuring Mistakes with a Loss Function

For one row, we need a number that says how wrong the prediction is. We will use:

```text
error_i = prediction_i - actual_i = y_hat_i - actual_i
```

Day 05 used the opposite sign and called it a **residual**:

```text
residual_i = actual_i - prediction_i
```

For squared quantities the sign disappears, so either convention works as long
as we use it consistently. Gradient descent formulas are usually written with
`error = prediction - actual`, so we use that here.

If we simply averaged the errors, positive and negative mistakes would cancel. A
model that predicts 10,000 too high for one house and 10,000 too low for another
would look perfect. To avoid that, we **square** each error.

The **mean squared error** (MSE) is:

```text
J(theta) = (1/n) * sum( (y_hat_i - y_i)^2 )
```

`J` is called the **loss** or **cost**. The subscript `i` means "do this for every
training row". The model's goal is to find the parameters that make `J` small.

Some books use the **sum** of squared errors instead of the mean:

```text
SSE(theta) = sum( (y_hat_i - y_i)^2 )
```

`MSE = SSE / n`, so minimizing one minimizes the other. The mean is easier to
compare across datasets of different sizes. MSE still has squared units (dollars
squared for house prices), which is why Day 05 also introduced RMSE for reporting.

### A tiny example

Take three points: `(1, 2)`, `(2, 4)`, and `(3, 5)`. Candidate line:
`y_hat = 1.5x + 0.5`.

| x | Actual `y` | Prediction `y_hat` | Error `y_hat - y` | Squared error |
|--:|-----------:|-------------------:|------------------:|--------------:|
| 1 | 2 | 2.0 | 0.0 | 0.00 |
| 2 | 4 | 3.5 | -0.5 | 0.25 |
| 3 | 5 | 5.0 | 0.0 | 0.00 |

```text
SSE = 0.00 + 0.25 + 0.00 = 0.25
MSE = 0.25 / 3 ≈ 0.083
```

Squaring does two useful things, as we saw in Day 05:

1. Every error becomes positive, so mistakes cannot cancel.
2. Large errors count much more than small ones. An error of 20 costs four times
   as much as an error of 10, not twice.

### Why MSE is a friendly loss

For a linear model, MSE is **convex** in the parameters. Picture the loss as a
bowl: it has one bottom, and there are no separate dips that could trap a search.
That is exactly why gradient descent is reliable here. For some other models the
loss surface can have many valleys, and the same algorithm may get stuck in a
poor one.

## 4. The Loss Surface

Think of every possible pair `(b, w)` as a location on a map. The height of the
map at that location is the loss `J(b, w)`. We start somewhere (often `b = 0`,
`w = 0`) and want to walk downhill to the lowest point.

For a single parameter, the picture is a curve. The **slope of the curve** at the
current point tells us which way is uphill. For two parameters, the loss is a
bowl-shaped surface, and the slope depends on which direction we move.

- If moving `w` slightly to the right **increases** the loss, we should move left.
- If moving `w` slightly to the right **decreases** the loss, we should move right.

The derivative gives us exactly that information, for each parameter separately.

## 5. Derivatives and the Gradient in Plain Language

A **derivative** measures how quickly one quantity changes when another changes a
little. For a function of several parameters we use a **partial derivative**:
differentiate with respect to one parameter while holding the others fixed.

For our loss `J(b, w)` there are two partial derivatives:

```text
dJ/db  -> how the loss changes when b changes a little
dJ/dw  -> how the loss changes when w changes a little
```

Together they form the **gradient**, written `grad J` or `∇J`:

```text
gradient = [ dJ/db, dJ/dw ]
```

The gradient points in the direction of **steepest increase** in the loss. So the
direction of **steepest decrease** is the negative gradient. That single sentence
is the whole idea behind gradient descent.

A useful sanity check:

| If the partial derivative is... | Increasing that parameter... | What to do |
|---------------------------------|------------------------------|------------|
| positive | increases the loss | decrease the parameter |
| negative | decreases the loss | increase the parameter |
| zero | does not change the loss locally | you may be at the bottom |

The update rule below does all three cases at once.

## 6. Deriving the Gradients for a Straight Line

We start with one feature. For row `i`:

```text
y_hat_i = b + w * x_i
error_i = y_hat_i - y_i = b + w * x_i - y_i
J(b, w) = (1/n) * sum( error_i^2 )
```

To differentiate a squared error, we use the **chain rule**: differentiate the
square first, then multiply by the derivative of what is inside.

```text
d(error_i^2) / d(theta) = 2 * error_i * d(error_i) / d(theta)
```

### Derivative with respect to the intercept `b`

Inside the error, `b` appears with coefficient `1`, so:

```text
d(error_i) / db = 1
d(error_i^2) / db = 2 * error_i * 1 = 2 * error_i
```

Averaging over all `n` rows:

```text
dJ/db = (2/n) * sum( error_i )
```

### Derivative with respect to the slope `w`

Inside the error, `w` is multiplied by `x_i`, so:

```text
d(error_i) / dw = x_i
d(error_i^2) / dw = 2 * error_i * x_i
```

Averaging over all rows:

```text
dJ/dw = (2/n) * sum( error_i * x_i )
```

### The result in code-friendly form

Since `error_i = prediction_i - actual_i`:

```text
gradient_b = (2/n) * sum( prediction_i - actual_i )
gradient_w = (2/n) * sum( (prediction_i - actual_i) * x_i )
```

The factor `2` comes from differentiating the square. If we had defined the loss
with an extra `1/2`, as some books do:

```text
J = (1/(2n)) * sum( error_i^2 )
```

then the `2` would cancel and the gradients would simply be the averages. Either
convention is correct if used consistently. The practical uses the version with
the `2`, so keep it.

### Several features at once

With a feature matrix `X` (rows are examples, columns are features, plus a column
of ones for the intercept) and parameter vector `theta`:

```text
predictions = X * theta
errors      = predictions - y
gradient    = (2/n) * X^T * errors
```

`X^T` means "transpose X": it turns the columns of errors back into one component
per parameter. This compact formula is exactly what the NumPy implementation
uses. It is the same arithmetic as the two formulas above, just organized for
many parameters.

## 7. From Gradient to Update

The gradient points uphill, so we move the parameters a small step in the
opposite direction:

```text
b = b - alpha * gradient_b
w = w - alpha * gradient_w
```

- `alpha` is the **learning rate**. It controls how large each step is.
- The minus sign is what makes us go **down** the loss surface.

For many parameters, the rule is applied to every component:

```text
theta_j = theta_j - alpha * (dJ/dtheta_j)    for every j
```

The updates should be computed from the **same** predictions and applied at the
same time. If you update `b` first and then use the new `b` to compute the `w`
gradient, you are no longer following the gradient of the original point. In code
this is easy: compute every gradient, then update the whole vector at once.

### One update with numbers

Suppose the training set has only two rows: `x = 1, y = 2` and `x = 2, y = 4`.
Start at `b = 0`, `w = 0`, and use `alpha = 0.1`.

First prediction and error for each row:

```text
row 1: y_hat = 0 + 0*1 = 0,  error = 0 - 2 = -2
row 2: y_hat = 0 + 0*2 = 0,  error = 0 - 4 = -4
```

With `n = 2`:

```text
MSE        = ( (-2)^2 + (-4)^2 ) / 2 = (4 + 16) / 2 = 10
gradient_b = (2/2) * (-2 + -4) = -6
gradient_w = (2/2) * (-2*1 + -4*2) = -10
```

Update:

```text
new b = 0 - 0.1 * (-6) = 0.6
new w = 0 - 0.1 * (-10) = 1.0
```

Now repeat. The table shows the state at the start of each pass, the gradients
computed there, and the resulting update.

| Pass | b | w | Errors | MSE | gradient_b | gradient_w | Next b | Next w |
|-----:|--:|--:|--------|----:|-----------:|-----------:|-------:|-------:|
| 0 | 0.00 | 0.00 | -2.00, -4.00 | 10.000 | -6.00 | -10.00 | 0.60 | 1.00 |
| 1 | 0.60 | 1.00 | -0.40, -1.40 | 1.060 | -1.80 | -3.20 | 0.78 | 1.32 |
| 2 | 0.78 | 1.32 | +0.10, -0.58 | 0.173 | -0.48 | -1.06 | 0.828 | 1.426 |

The loss falls from 10 to about 0.17 in two updates. The true line for these
points is `y = 2x`, which means `b = 0` and `w = 2`. The estimates are moving in
the right direction, and the loss is dropping. More passes would bring them
closer.

Real datasets have many rows, noise, and several features, so one or two updates
are never enough. The algorithm repeats the same four operations thousands of
times.

## 8. The Complete Batch Gradient Descent Algorithm

**Batch** gradient descent uses all training rows to compute each update.

1. Choose starting values for the parameters, often all zero.
2. Predict every training row with the current parameters.
3. Compute the error for every row: `prediction - actual`.
4. Compute one gradient component per parameter using all rows.
5. Update every parameter by subtracting `alpha * gradient`.
6. Repeat steps 2–5 until the loss stops improving or a maximum number of
   iterations is reached.

In pseudocode for one feature:

```text
initialize b = 0, w = 0

repeat until stopping condition:
    predictions = b + w * x          # for all rows
    errors      = predictions - y
    gradient_b  = (2/n) * sum(errors)
    gradient_w  = (2/n) * sum(errors * x)
    b = b - alpha * gradient_b
    w = w - alpha * gradient_w
```

This is exactly the `fit_gradient_descent` function used in the practical.

### Variants of gradient descent

| Variant | Rows per update | Property | When it is useful |
|---------|-----------------|----------|-------------------|
| Batch gradient descent | all `n` rows | stable direction, one update per full pass | small to medium datasets |
| Stochastic gradient descent (SGD) | 1 row | noisy, very cheap updates | huge datasets, online learning |
| Mini-batch gradient descent | a small batch, e.g. 32 or 256 | middle ground, hardware-friendly | most deep learning and large-scale work |

Batch descent follows the true gradient smoothly. SGD jumps around but can make
progress before seeing the whole dataset, and the noise can even help escape poor
regions in non-convex problems. Today we implement the batch version because it
is the easiest to check by hand.

### When does it stop?

Common stopping rules:

- a fixed maximum number of iterations;
- the loss changes by less than a tiny threshold;
- the gradient becomes very small;
- a validation score stops improving (the safest choice in a real project).

A maximum iteration count is a safety net. Always keep one so a bad learning rate
cannot run forever.

### Cost per iteration

Each batch iteration touches every row and every feature, so the cost grows with
`n * p`. That is cheap when the data fits in memory. When it does not, mini-batch
gradient descent is the usual answer.

## 9. Learning Rate and Convergence

The learning rate `alpha` controls the step size and is the most important knob
in gradient descent.

| Learning rate | Typical behavior | What you see in the loss curve |
|---------------|------------------|--------------------------------|
| Too small | Tiny steps; learning is very slow | Smooth but almost flat |
| Sensible | Steady progress to the bottom | Falls quickly, then flattens |
| Too large | Steps overshoot the bottom | Bounces, spikes, or grows |
| Far too large | Diverges | Loss becomes huge or `NaN` |

Here are actual results from the practical's toy problem (standardized input,
1,000 iterations). The first column is the loss before any update; the second is
the loss at the end:

| Learning rate | Starting MSE | Final MSE (1,000 iterations) | Reading |
|---------------|-------------:|-----------------------------:|---------|
| 0.001 | 193.9 | 4.15 | still far from the bottom |
| 0.01 | 193.9 | 0.61 | converged, but slowly |
| 0.1 | 193.9 | 0.61 | converged quickly |
| 0.5 | 193.9 | 0.61 | converged very quickly |
| 1.0 | 193.9 | 193.9 | oscillates; no progress |

These exact numbers depend on the data and the scaling. The **pattern** is what
matters: small steps are slow, a good value converges, and too large a value
either bounces or diverges.

Why can a value of `1.0` fail? After standardization, the two columns are
orthogonal and have similar scale, so the safe range for this particular toy
problem is roughly `alpha < 1`. A value of exactly `1.0` makes the parameters
flip back and forth between two points without settling. With real, correlated
features the safe upper limit is usually smaller, which is why scaling matters.

### Convergence is not the same as a good model

When the loss stops changing much, we say the algorithm has **converged**. This
only means it found a low training loss. It does **not** prove the model predicts
well on new data. We still evaluate on held-out rows, exactly as in Day 05.

Also, the training loss curve can keep creeping down slightly forever because of
noise. Stop when the improvement is negligible, not only when it is exactly zero.

## 10. Feature Scaling

### The problem: features live on different scales

Imagine predicting house price from:

- `area_sqft`, roughly 600 to 3,000;
- `bedrooms`, roughly 1 to 5;
- `age_years`, roughly 0 to 40.

The gradient for each weight is multiplied by its feature. A change of one unit
in `area_sqft` is tiny, while a change of one unit in `bedrooms` is large relative
to its range. The loss surface becomes a long, thin valley: steep in one direction
and almost flat in another. Gradient descent then zigzags, needs a very small
learning rate, and converges slowly.

### Two common transforms

**Standardization** (also called z-score scaling):

```text
z = (value - mean) / standard_deviation
```

After this, the feature has mean 0 and standard deviation 1. It keeps the shape
of the distribution and does not bound the values.

**Min-max scaling**:

```text
scaled = (value - min) / (max - min)
```

After this, every value lies between 0 and 1. It is sensitive to outliers, because
one extreme value changes `min` or `max`.

| Method | Formula | Resulting range | Good when |
|--------|---------|-----------------|-----------|
| Standardization | `(x - mean) / std` | centered at 0, not bounded | features are roughly bell-shaped or have outliers |
| Min-max | `(x - min) / (max - min)` | exactly 0 to 1 | you need a bounded range and outliers are not a concern |

The practical uses standardization because it pairs well with gradient descent and
regularization.

### Fit on training data only

The mean, standard deviation, minimum, and maximum are **learned from the training
set**, then reused on the test set:

```text
fit the scaler on X_train
transform X_train
transform X_test with the same mean and standard deviation
```

If you compute the mean or standard deviation from the full dataset, information
from the test rows leaks into training. That is **data leakage** from Day 04. It
can make your test score look better than it really is.

### What scaling does and does not do

- For ordinary least squares, scaling does **not** change the predictions. It only
  changes the units of the coefficients.
- For gradient descent, scaling makes the steps better balanced and allows a
  larger, safer learning rate.
- For Ridge and Lasso, scaling puts every penalty on the same footing. Without it,
  a feature measured in small units can be penalized unfairly.
- Scaled coefficients are **not** in the original units. A weight on a
  standardized feature means "change in prediction per one standard deviation",
  not "per one square foot".

Models that rely on distances or gradients usually need scaling: gradient descent,
Ridge, Lasso, k-nearest neighbors, support vector machines, PCA, and neural
networks. Tree-based models are mostly insensitive to monotonic rescaling.

## 11. Polynomial Features

A straight line cannot follow a curved pattern. But we can give the model a
curved input. For one feature, add its square:

```text
y_hat = b + w1*x + w2*x^2
```

This is still a linear model in its parameters: the prediction is a weighted sum
of known columns. The columns are now `x` and `x^2`, not just `x`. Least squares
and gradient descent work exactly as before; only the feature matrix changed.

For two features `x1` and `x2`, a degree-2 expansion adds:

| Term | Name | Meaning |
|------|------|---------|
| `x1` | original | straight effect of feature 1 |
| `x2` | original | straight effect of feature 2 |
| `x1^2` | square | curvature in feature 1 |
| `x1*x2` | interaction | effect of feature 1 depends on feature 2 |
| `x2^2` | square | curvature in feature 2 |

The interaction term is often the most valuable addition in real problems,
because it lets the effect of one feature change with another.

### Pipeline order

When combining polynomial features with scaling:

```text
PolynomialFeatures -> StandardScaler -> model
```

Expand first, then scale the expanded columns. If you scale first, the squares
and interactions are created from already-scaled values, which is usually not what
you want. A scikit-learn `Pipeline` applies the steps in order and fits each one
on the training data only.

### The danger: overfitting

Polynomial features make the model more flexible. More flexibility can fit the
training data better while making test predictions worse. On the house data used
in the practical, with the same split:

| Polynomial degree | Train RMSE | Test RMSE |
|-------------------|-----------:|----------:|
| 1 (straight line) | 12,645 | 15,006 |
| 2 | 9,030 | 9,991 |
| 3 | 9,007 | 10,144 |
| 5 | 8,927 | 10,200 |

Degree 2 clearly helps. Going from degree 2 to degree 5 improves the **training**
error slightly but makes the **test** error worse. That is overfitting: the extra
wiggles are fitting noise.

Two more cautions:

- **Extrapolation:** high-degree polynomials can swing wildly outside the range of
  the training data. A degree-5 curve may look fine between 600 and 3,000 square
  feet and then produce absurd prices at 5,000.
- **Interpretation:** coefficients on `x`, `x^2`, and `x1*x2` do not have the
  simple "one more unit" meaning of a plain linear coefficient.

## 12. Multicollinearity and Regularization

### Multicollinearity

**Multicollinearity** means two or more features carry almost the same
information. Common causes:

- including `area` and `area_sqft` in the same model;
- including a total and its parts (total rooms and bedrooms plus bathrooms);
- adding polynomial terms that are highly correlated with the original feature.

Polynomial features are a natural example: on the house data, `area` and `area^2`
have a correlation of about **0.99**. When features are nearly duplicates, the
model can trade a large positive weight on one against a large negative weight on
the other and still fit the data. The predictions may be fine, but the
coefficients become unstable and hard to interpret, and the loss surface becomes
ill-conditioned, which slows gradient descent.

Regularization controls this by adding a penalty for large weights to the loss.

### Ridge regression (L2 penalty)

Ridge adds the sum of **squared** weights:

```text
J_ridge = MSE + alpha * sum( w_j^2 )
```

- The penalty grows quickly for large weights, so Ridge shrinks them.
- It keeps all features; weights become smaller but usually not exactly zero.
- It is especially helpful when features are correlated, because it spreads the
  weight across them instead of letting them cancel wildly.
- `alpha` (called `lambda` in many textbooks) controls the strength.

### Lasso regression (L1 penalty)

Lasso adds the sum of **absolute** weights:

```text
J_lasso = MSE + alpha * sum( |w_j| )
```

- It also shrinks weights, but the shape of the penalty can push some weights
  **exactly to zero**.
- That makes Lasso a **feature selection** method: the features with zero weight
  are effectively dropped.
- With many candidate features, Lasso can produce a simpler, more interpretable
  model.

### Elastic Net

**Elastic Net** combines both penalties:

```text
J_elastic = MSE + alpha * ( rho * sum(|w_j|) + (1 - rho) * sum(w_j^2) )
```

It is a useful compromise when features are correlated (Ridge-like behavior) but
you also want some sparsity (Lasso-like behavior).

### The size of the penalty matters

On the house data with the quadratic pipeline, using the same split:

| Ridge `alpha` | Train RMSE | Test RMSE | Reading |
|---------------|-----------:|----------:|---------|
| 0.01 | 9,030 | 9,997 | almost no penalty |
| 0.1 | 9,034 | 10,048 | very light penalty |
| 1 | 9,196 | 10,517 | mild shrinkage |
| 10 | 10,064 | 12,262 | noticeably flatter fit |
| 100 | 19,900 | 23,075 | underfitting |
| 1,000 | 46,509 | 48,558 | nearly a flat line |

- A penalty that is too weak does little.
- A penalty that is too strong makes the model too simple, the opposite problem.
- The best value depends on the data and must be chosen on **training** data with
  cross-validation, not by repeatedly checking the test set.

### Ridge versus Lasso on duplicated features

The different penalties have different personalities. Suppose we accidentally add
the same feature twice (`area_sqft` and an exact copy). With standardized
features:

| Model | Weight on feature 1 | Weight on feature 2 | Behavior |
|-------|--------------------:|--------------------:|----------|
| Ridge (`alpha=10`) | 26,317 | 26,317 | splits the weight evenly |
| Lasso (`alpha=1000`) | 53,985 | 0 | keeps one, drops the other |

Ridge shares credit. Lasso picks one and discards the duplicate. Neither behavior
is "wrong"; they answer different questions.

### Practical rules for regularization

1. Standardize features before Ridge or Lasso, so the penalty is fair.
2. The intercept is normally **not** penalized; scikit-learn handles this.
3. Choose `alpha` with cross-validation on the training set.
4. Report coefficients carefully: they are on the scaled features and include the
   effect of shrinkage.
5. Regularization reduces variance at the cost of some bias. That is the
   bias-variance tradeoff from Day 04 in action.

> Different libraries define `alpha` relative to the mean squared error or the sum
> of squared errors. The values used in this course come from scikit-learn, whose
> Ridge and Lasso objectives are based on the **sum** of squared errors. A useful
> `alpha` is therefore dataset- and library-dependent; read the documentation
> rather than copying a number from another project.

## 13. Evaluating the Models You Build

The metrics from Day 05 still apply:

- **MAE:** average absolute error, in the target's units.
- **RMSE:** square root of MSE, also in the target's units, punishing large errors
  more.
- **R²:** fraction of variation explained, compared with always predicting the
  mean.

Two rules keep the comparison honest:

1. **Use the same train/test split for every model.** Otherwise differences may
   come from the split rather than the model.
2. **Compare against a baseline.** For regression, the simplest baseline predicts
   the training mean. On the house split used in the practical, that baseline has
   a test RMSE of about **58,015** dollars.

Results from the practical (same split, 28 test houses):

| Model | Test MAE | Test RMSE | Test R² |
|-------|---------:|----------:|--------:|
| Baseline (predict the mean) | 50,997 | 58,015 | 0.000 |
| Linear regression | 13,199 | 15,006 | 0.933 |
| Quadratic regression | 7,866 | 9,991 | 0.970 |
| Quadratic Ridge (`alpha=10`) | 10,804 | 12,262 | 0.955 |
| Quadratic Lasso (`alpha=0.1`) | 7,866 | 9,991 | 0.970 |

Reading the table:

- Every real model beats the baseline, so they are learning something.
- The quadratic model improves on the straight line, which matches the curved
  pattern we see in the scatter plot.
- Ridge at `alpha=10` is worse than the unpenalized quadratic model **here**,
  because that penalty is too strong for this small dataset.
- Lasso at `alpha=0.1` behaves almost like ordinary regression, because the
  penalty is too weak to zero either of the two features.

Do not tune `alpha` by staring at this test table. In a real project, split the
training data again into training and validation folds, choose the penalty there,
and touch the test set once at the end. Day 19 and Day 20 cover cross-validation
and pipelines for exactly this reason.

## 14. Real-World Use Case: Forecasting Store Sales

A retail chain wants to predict next month's sales for each store.

- **Row:** one store for one month.
- **Features:** floor area, number of employees, local population, competitor
  count, promotion spending, month of year, previous-month sales.
- **Target:** sales in dollars.
- **Metrics:** MAE for "on average, how far off are we in dollars?" and RMSE
  because large misses are expensive to plan around.

Why the ideas from today matter here:

- **Gradient descent / mini-batch:** there may be thousands of stores and years of
  history, and the model may be retrained as new months arrive. Mini-batch
  gradient descent handles that comfortably.
- **Scaling:** floor area is in thousands of square feet, employee counts are
  small, and promotion spending is in dollars. Without scaling, optimization is
  slow and penalties are unfair.
- **Polynomial and interaction features:** sales may grow faster for very large
  stores, and promotion spending may matter more during holidays. An interaction
  such as `promotion_spending * holiday_month` can capture that.
- **Ridge:** "previous-month sales" and "previous-quarter sales" are highly
  correlated. Ridge stabilizes their coefficients.
- **Lasso:** when the team has collected fifty candidate features, Lasso can drop
  the useless ones and leave a shorter, easier-to-explain model.

What could go wrong:

- **Leakage:** using this month's sales to predict this month's target, or
  computing scaling statistics on future months.
- **Distribution shift:** a new competitor opens, so last year's pattern no longer
  holds; monitor the model after deployment.
- **Extrapolation:** a store twice as large as any in the training data may get a
  wild polynomial prediction.
- **Overfitting:** dozens of polynomial terms can fit historical noise; compare on
  held-out months and regularize.
- **Causation:** a coefficient on promotion spending describes an association, not
  proof that raising spending will raise sales by that amount.

The workflow is the same as Day 04 and Day 05: define the problem, split by time,
fit preprocessing on training data only, compare with a baseline, evaluate on
unseen data, and monitor after deployment.

## 15. Common Mistakes

1. Forgetting to scale features before gradient descent or regularization.
2. Computing the scaler's mean or standard deviation on the full dataset (leakage).
3. Choosing a learning rate that is far too large, then blaming the model.
4. Choosing a learning rate that is far too small and stopping before convergence.
5. Updating parameters one at a time instead of computing all gradients first.
6. Mixing loss conventions: using a `1/2` in the loss but not in the gradient, or
   the reverse.
7. Watching only the training loss and never checking the test set.
8. Adding high-degree polynomial terms and trusting the improved training error.
9. Interpreting coefficients on standardized or polynomial features as if they
   were raw-unit effects.
10. Applying Ridge or Lasso to unscaled features.
11. Selecting `alpha` by repeatedly scoring the test set.
12. Extrapolating a polynomial far outside the training range.
13. Assuming convergence means the model generalizes.
14. Ignoring multicollinearity and then over-interpreting unstable coefficients.
15. Using a linear model on a target that is a category; that is Day 07's job.

## 16. Putting It Together

Before trusting a gradient-descent or regularized regression model, ask:

1. Is the target a number, not a category?
2. Did I split the data before fitting any preprocessing?
3. Are features scaled using statistics from the training set only?
4. Is the learning rate small enough for the loss to fall, but large enough to
   converge in reasonable time?
5. Does the loss curve flatten, rather than bounce or explode?
6. Did I compute all gradients before applying the update?
7. Does my gradient-descent result agree (roughly) with `LinearRegression`?
8. Does the model beat the "predict the mean" baseline on unseen data?
9. Did I compare straight, polynomial, and regularized versions on the same split?
10. Is the polynomial degree justified by the test score, not just the training
    score?
11. Did I choose `alpha` with cross-validation rather than the test set?
12. Am I staying inside the range of the training data when predicting?

If the answers are reasonable, you have a model you can defend.

## 17. Summary

```text
predict -> compute errors -> compute gradients -> update parameters -> repeat
```

- The gradient is the direction of steepest **increase** in loss.
- Subtracting a scaled gradient moves parameters toward lower loss.
- The learning rate controls step size and therefore convergence.
- Feature scaling makes the loss surface easier to navigate and makes penalties
  fair.
- Polynomial features let a linear model represent curves and interactions, at the
  risk of overfitting.
- Ridge shrinks weights; Lasso can shrink some weights to exactly zero.
- Convergence is about optimization; generalization is about unseen data. Check
  both.

## Outcome

Learners can explain why gradient descent works, derive the intercept and slope
gradients for squared-error linear regression, perform an update by hand, describe
batch versus stochastic descent, explain how learning rate and scaling affect
convergence, create polynomial features, and compare ordinary, Ridge, and Lasso
regression with honest held-out evaluation.
