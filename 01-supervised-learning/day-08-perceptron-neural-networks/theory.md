# Day 08 — Theory: Perceptron and Neural Network Fundamentals

**Module:** Module 2: Supervised Learning

For seven days we have built models that are **linear in their inputs**, or linear
in features we engineered by hand. Logistic regression draws a line (or a
hyperplane). Ridge and Lasso shrink that line. Even the sigmoid only bends the
*output* of a linear score; it does not bend the boundary in feature space.

Many real patterns are not linear. A face is not a weighted sum of pixel values. A
spoken word is not a straight line through audio samples. To reach these patterns,
we need models that can **compose** simple pieces into complicated ones. That is the
idea behind a **neural network**.

The story starts in 1958 with the **perceptron**, the first trainable artificial
neuron. It is elegant, it has a convergence proof, and it fails on a tiny problem
called XOR. That failure, and the fix, is the bridge from classical ML to modern
deep learning.

## In Plain English

**Think of it like this:** a neuron is one tiny decision-maker; a neural network
is a team of them wired in layers, each layer passing useful signals upward.

- A neuron does a **weighted sum + bias**, then applies an **activation**.
- One perceptron can only draw a straight boundary; **XOR** shows its limit.
- **Hidden layers** build new features and allow curved boundaries.
- **Forward propagation** makes a prediction; **backpropagation** sends the
  blame backward to adjust the weights.
- Deep networks need more data and compute; on small tables, classical models
  often match them.

## 1. From a Line to a Network

A linear model computes one score:

```text
z = b + w1*x1 + w2*x2 + ... + wp*xp
```

A neural network computes many scores, feeds them into other units, and combines
those. Each unit is itself a tiny linear model followed by a nonlinearity:

```text
unit:      z = w·x + b
output:    a = f(z)
```

The nonlinearity `f` is called the **activation function**. Stacking units with
nonlinear activations lets the network represent curved, interacting, and
hierarchical patterns.

| Era | Model | Boundary | Learns |
|-----|-------|----------|--------|
| 1958 | Perceptron | linear | weights, from mistakes |
| 1960s | Logistic regression | linear | weights, by maximum likelihood |
| 1986 | Multi-layer network + backpropagation | nonlinear | many layers of weights |
| 2010s | Deep networks | highly nonlinear | representations from raw data |

The perceptron and logistic regression are both **single neurons**. A neural
network is what happens when we connect many of them.

## 2. The Artificial Neuron

An **artificial neuron** takes a vector of inputs, multiplies each by a weight, adds
a bias, and passes the result through an activation.

```text
z = w1*x1 + w2*x2 + ... + wp*xp + b
a = f(z)
```

In vector form:

```text
z = w · x + b
a = f(z)
```

| Symbol | Meaning |
|--------|---------|
| `x` | input vector, one row of features |
| `w` | weight vector, learned |
| `b` | bias, learned |
| `z` | weighted sum, also called the pre-activation |
| `f` | activation function |
| `a` | activation, the unit's output |

### What the weights and bias do

- A **weight** says how strongly an input pushes the unit. A large positive weight
  means "more of this input makes the unit fire more". A negative weight means the
  opposite.
- The **bias** shifts the firing threshold. It is a weight attached to a constant
  input of 1. Without a bias, the boundary must pass through the origin; with a
  bias, the unit can fire at any input level.

For a step activation, the unit fires when:

```text
w · x + b >= 0
```

That is a **hyperplane** in feature space. The weights set its orientation, and the
bias sets its offset.

### A neuron is logistic regression with a different activation

If `f` is the sigmoid, the neuron is exactly logistic regression from Day 07:

```text
a = sigma(w · x + b) = P(y = 1 | x)
```

If `f` is a step function, the neuron is the classic perceptron. If `f` is the
identity, the neuron is linear regression. The neuron is a template; the activation
chooses the behavior.

### A single neuron by hand

Take `x = [1, 2]`, `w = [0.5, -0.5]`, and `b = 0`:

```text
z = 0.5*1 + (-0.5)*2 + 0 = -0.5
step(z)    = 0
sigmoid(z) = 0.378
```

The same weighted sum becomes a hard label with a step, a probability with a
sigmoid, or a raw number with the identity. The inputs and weights are identical;
only the activation changes.

## 3. Activation Functions

The activation is where the nonlinearity enters. Without it, a deep network would
collapse into a single linear model.

| Name | Formula | Output range | Derivative | Typical use |
|------|---------|--------------|------------|-------------|
| Identity | `f(z) = z` | `(-inf, inf)` | `1` | regression output |
| Step | `1 if z >= 0 else 0` | `{0, 1}` | `0` almost everywhere | classic perceptron |
| Sigmoid | `1 / (1 + exp(-z))` | `(0, 1)` | `f(z) * (1 - f(z))` | binary output |
| Tanh | `(exp(z) - exp(-z)) / (exp(z) + exp(-z))` | `(-1, 1)` | `1 - f(z)^2` | hidden layers in shallow nets |
| ReLU | `max(0, z)` | `[0, inf)` | `0` if `z < 0`, else `1` | hidden layers in modern nets |
| Softmax | `exp(z_k) / sum_j exp(z_j)` | `(0, 1)`, sums to 1 | vector Jacobian | multiclass output |

### Step

The step function is a hard switch. It is perfect for a decision and useless for
gradient descent: its slope is zero almost everywhere and undefined at the jump.
The perceptron gets around this with a custom update rule instead of a gradient.

### Sigmoid

The sigmoid is smooth and bounded. It turns any score into a probability. It was
the standard hidden activation for decades, but it has two problems when stacked:

- its output is always positive, so it is not zero-centered;
- its derivative is at most `0.25`, so gradients shrink as they pass backward
  through many layers.

That shrinking is the **vanishing gradient** problem, discussed in Section 9.

### Tanh

Tanh is a shifted and scaled sigmoid:

```text
tanh(z) = 2 * sigmoid(2z) - 1
```

It ranges from `-1` to `1`, and it is zero-centered, which usually trains better
than the sigmoid. Its derivative is at most `1`, at `z = 0`, but it still saturates
in the tails.

### ReLU

The rectified linear unit is the default hidden activation in modern networks:

```text
ReLU(z) = max(0, z)
```

- It is cheap to compute.
- Its derivative is exactly `1` for positive inputs, so gradients do not vanish on
  the active side.
- It is sparse: roughly half the units output zero for a given input.

Its weakness is the **dying ReLU**: a unit whose inputs are always negative has zero
gradient and stops learning. Variants such as Leaky ReLU and ELU address this.

### Softmax

Softmax turns a vector of scores into a probability distribution over classes:

```text
P(class k | x) = exp(z_k) / sum_j exp(z_j)
```

It is the multiclass counterpart of the sigmoid. Use softmax with categorical
cross-entropy, not with squared error.

### Why nonlinearity is essential

Suppose every layer is linear. Then the whole network is linear, no matter how many
layers it has:

```text
A2 = W2 * (W1 * X + b1) + b2
   = (W2 * W1) * X + (W2 * b1 + b2)
   = W * X + b
```

Two linear layers are one linear layer in disguise. The only way depth buys
representational power is by inserting a nonlinear activation between layers. This
single fact explains why the activation function is not a minor detail.

## 4. The Perceptron

The **perceptron** (Rosenblatt, 1958) is the simplest trainable neuron. It uses a
step activation and learns from its mistakes.

### The prediction rule

For an input `x` and label `y` in `{0, 1}`:

```text
y_hat = 1 if w · x + b >= 0 else 0
```

### The learning rule

Compare the prediction with the truth, then nudge the weights in the direction that
would have been more correct:

```text
w <- w + learning_rate * (y - y_hat) * x
b <- b + learning_rate * (y - y_hat)
```

Because `y - y_hat` is `0`, `+1`, or `-1`, there are only three cases:

| True `y` | Predicted `y_hat` | `y - y_hat` | Effect |
|---------:|------------------:|------------:|--------|
| 0 | 0 | 0 | correct, no change |
| 1 | 1 | 0 | correct, no change |
| 1 | 0 | +1 | push `w` toward `x`, raise `b` |
| 0 | 1 | -1 | push `w` away from `x`, lower `b` |

The perceptron is **mistake-driven**: correctly classified rows leave the model
untouched.

### One update by hand

Take `x = [1, 1]`, `y = 0`, start at `w = [0, 1]`, `b = 0`, and use a learning rate
of `1.0`.

```text
z = 0*1 + 1*1 = 1, y_hat = 1
error = y - y_hat = -1
w <- [0, 1] + 1.0 * (-1) * [1, 1] = [-1, 0]
b <- 0 + 1.0 * (-1) = -1
```

After the update, `z = -1*1 + 0*1 - 1 = -2 < 0`, so the unit now predicts `0` for
this row. A single update fixed a single example. The real algorithm repeats over
the whole dataset many times.

### The perceptron criterion

The update rule can be read as gradient descent on the **perceptron loss**:

```text
loss = max(0, -y * (w · x + b))     # with y in {-1, +1}
```

The loss is zero when the row is correctly classified with a margin, and grows
linearly when it is on the wrong side. This is why the perceptron only moves when
it makes a mistake.

### The convergence theorem

If the training data is **linearly separable**, and the learning rate is positive,
the perceptron makes only a finite number of mistakes and stops changing. A more
precise version says the number of mistakes is bounded by a quantity that grows with
the squared radius of the data and shrinks with the squared margin between classes.

Two consequences:

- On separable data, the perceptron is guaranteed to find *a* separating hyperplane.
- On non-separable data, it never settles; it keeps cycling and the mistake count
  per epoch does not reach zero.

The theorem does **not** say the result is unique, optimal, or robust. Different
starting points and different row orders give different separating lines.

### Perceptron versus logistic regression

| Property | Perceptron | Logistic regression |
|----------|------------|---------------------|
| Activation | step | sigmoid |
| Output | hard label | probability |
| Loss | perceptron criterion | log loss |
| Updates | only on mistakes | on every row |
| Separable data | guaranteed to converge | converges to a finite solution |
| Overlapping data | never settles | settles at a soft boundary |
| Calibration | none | probabilistic |

Both are linear classifiers. Logistic regression gives you probabilities and a
principled loss; the perceptron gives you a clean historical stepping stone.

## 5. Linear Separability and XOR

### Linear separability

A binary dataset is **linearly separable** if a single hyperplane can put all
positive rows on one side and all negative rows on the other. In two dimensions,
that means a straight line with no points on the wrong side.

Examples:

- Two well-separated blobs: separable.
- Two concentric rings: not separable.
- XOR: not separable.

When data is not separable, a linear classifier can still be useful; it just cannot
be perfect. The question is whether the boundary it finds is good enough.

### The XOR problem

XOR ("exclusive or") outputs 1 when exactly one input is 1:

| `x1` | `x2` | `y` |
|-----:|-----:|----:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Plot the four points. The two class-0 points are at opposite corners, and the two
class-1 points are at the other opposite corners. Any straight line can isolate one
corner from the other three, but no line puts both 0s on one side and both 1s on the
other.

### Why no perceptron can solve XOR

A perceptron predicts 1 exactly when:

```text
w1*x1 + w2*x2 + b >= 0
```

Assume a perceptron solves XOR. Then the four points force four inequalities:

```text
(0,0) -> 0:        b < 0
(1,1) -> 0:        w1 + w2 + b < 0
(0,1) -> 1:        w2 + b >= 0
(1,0) -> 1:        w1 + b >= 0
```

Add the two "predict 1" inequalities:

```text
w1 + w2 + 2b >= 0
```

Since `b < 0`, we can write this as `w1 + w2 + b >= -b > 0`. But the `(1,1)` row
requires `w1 + w2 + b < 0`. The two requirements contradict each other. Therefore no
such weights exist.

This is not a limitation of the learning algorithm. It is a limitation of the
**model class**: the boundary is a straight line, and the pattern is not.

### The historical impact

In 1969, Minsky and Papert published *Perceptrons*, showing that single-layer
perceptrons cannot represent XOR and similar functions. The result was widely read
as a verdict on the whole field, and neural network research cooled for years. The
fix, however, was already conceptually simple: add a **hidden layer**.

### How a hidden layer solves XOR

A two-layer network can first transform the inputs into new features, then classify
those. For example, compute OR and NAND in the hidden layer, then AND them:

```text
h1 = step(x1 + x2 - 0.5)          # OR
h2 = step(-x1 - x2 + 1.5)         # NAND
y  = step(h1 + h2 - 1.5)          # AND
```

Check all four points:

| `x1` | `x2` | `h1` (OR) | `h2` (NAND) | `y` (AND) |
|-----:|-----:|----------:|------------:|----------:|
| 0 | 0 | 0 | 1 | 0 |
| 0 | 1 | 1 | 1 | 1 |
| 1 | 0 | 1 | 1 | 1 |
| 1 | 1 | 1 | 0 | 0 |

The same four points are now separable, because the hidden layer mapped them into a
space where a line works. **Feature engineering became learned rather than
hand-written.** A network with a hidden layer is no longer a single linear
classifier.

As a shortcut, a single perceptron can also solve XOR if we manually add the
interaction feature `x1 * x2`, exactly as in Day 06. The hidden layer does this kind
of feature construction automatically.

## 6. Layers and Architecture

A **feedforward neural network** arranges neurons in layers:

- the **input layer** holds the features, one node per feature;
- one or more **hidden layers** transform the representation;
- the **output layer** produces the prediction.

"Feedforward" means information flows in one direction, from input to output, with
no cycles.

```text
input -> hidden -> hidden -> output
  x         a1       a2       y_hat
```

### Dense layers

In a **fully connected** (or **dense**) layer, every unit receives every output of
the previous layer. If a layer has `n_in` inputs and `n_out` units, it has:

```text
n_in * n_out weights + n_out biases
```

### Counting parameters

For the handwritten-digit network used in the practical, the architecture is:

```text
64 inputs -> 64 hidden units -> 10 output units
```

The parameter count is:

```text
layer 1: 64 * 64 + 64 = 4160
layer 2: 64 * 10 + 10 =  650
total:                    4810
```

Five thousand parameters is small by modern standards, but it is already far more
than a linear model with 640 weights. The extra capacity comes from the hidden
layer.

### Width and depth

- **Width** is the number of units in a layer. A wider layer can represent more
  features at one level.
- **Depth** is the number of layers. A deeper network can compose features
  hierarchically: edges, then parts, then objects, then concepts.

Depth is powerful because many functions have a compact deep representation and an
enormous shallow one. But depth also makes optimization harder, which is why modern
networks use ReLU, careful initialization, normalization, and residual connections.

### The universal approximation theorem

A feedforward network with a single hidden layer and enough units can approximate
any continuous function on a bounded region to arbitrary accuracy, under mild
conditions on the activation.

This is a statement about **representational capacity**, not about learning. It does
not say:

- that gradient descent will find those weights;
- that you have enough data to estimate them;
- that the network will generalize to new rows;
- that a single hidden layer is efficient.

In practice, depth and good data matter more than raw width.

## 7. Forward Propagation

**Forward propagation** is the left-to-right calculation that turns inputs into
predictions. For a two-layer network:

```text
A0 = X
Z1 = A0 * W1 + b1
A1 = f1(Z1)
Z2 = A1 * W2 + b2
A2 = f2(Z2)
```

Here `A0` is the input matrix, `W1` and `W2` are weight matrices, and `b1` and `b2`
are bias vectors broadcast across rows. Each row is one sample.

### A numeric example

Take one row `x = [1, 2]`, a hidden layer with two ReLU units, and these weights:

```text
W1 = [[0.5, -0.5],
      [1.0,  0.5]]
b1 = [0, 0]

W2 = [[ 1.0],
      [-1.0]]
b2 = [0]
```

Forward pass:

```text
Z1 = [1*0.5 + 2*1.0, 1*(-0.5) + 2*0.5] = [2.5, 0.5]
A1 = ReLU([2.5, 0.5]) = [2.5, 0.5]
Z2 = 2.5*1.0 + 0.5*(-1.0) = 2.0
```

If the task is regression, the output is `2.0`. If it is binary classification, pass
`2.0` through a sigmoid to get `0.881`.

### Output activation by task

| Task | Output units | Output activation | Loss |
|------|--------------|-------------------|------|
| Regression | 1 (or more) | identity | MSE or MAE |
| Binary classification | 1 | sigmoid | binary cross-entropy |
| Multiclass classification | `K` | softmax | categorical cross-entropy |
| Multilabel classification | one per label | sigmoid per label | binary cross-entropy per label |

Matching the output activation to the loss is not optional. A softmax with squared
error trains poorly; a sigmoid with softmax-style targets is simply wrong.

## 8. Loss Functions

A loss turns predictions and targets into one number that training tries to reduce.

### Regression

Mean squared error, the same as Day 05:

```text
J = (1/n) * sum( (y_hat_i - y_i)^2 )
```

The derivative with respect to the output is `2 * (y_hat - y) / n`; the factor of 2
is often folded into the learning rate.

### Binary classification

Binary cross-entropy, the same as Day 07:

```text
J = -(1/n) * sum( y_i*log(p_i) + (1 - y_i)*log(1 - p_i) )
```

Here `p_i` is the sigmoid output. It punishes confident wrong predictions heavily.

### Multiclass classification

Categorical cross-entropy, also called the multiclass log loss:

```text
J = -(1/n) * sum_i sum_k y_ik * log(p_ik)
```

For a true class `k`, `y_ik = 1` and all other entries are 0, so the inner sum
reduces to `-log(p_i,true)`. The loss cares only about the probability assigned to
the correct class.

### Why not just use accuracy?

Accuracy is a step function of the parameters: small weight changes rarely change
the number of correct predictions. Its gradient is zero almost everywhere, so
gradient descent cannot follow it. Accuracy is a good **reporting** metric and a bad
**training** loss. This is exactly the lesson from Day 07.

## 9. Backpropagation

**Backpropagation** is how a network computes the gradient of the loss with respect
to every weight. It is the chain rule applied efficiently, layer by layer, from the
output backward.

### The intuition

Forward propagation answers: "given these weights, what is the prediction?" The loss
then says how wrong that prediction is. Backpropagation answers: "how much did each
weight contribute to that error?"

A weight deep in the network affects the loss only through the units it feeds. So we
first compute the error at the output, then send a share of the blame backward to
each hidden unit, then to each earlier unit.

### The chain rule on a computational graph

Each operation is a node in a graph. For a two-layer network:

```text
X -> Z1 -> A1 -> Z2 -> A2 -> J
```

For any weight `w`, the chain rule says:

```text
dJ/dw = dJ/dA2 * dA2/dZ2 * dZ2/dA1 * dA1/dZ1 * dZ1/dw
```

Backpropagation computes these factors once per layer and reuses them, instead of
recomputing the full chain for every weight separately.

### The delta notation

Define the **error signal** at a layer as the derivative of the loss with respect to
that layer's pre-activation:

```text
delta^l = dJ / dZ^l
```

For the output layer with sigmoid plus binary cross-entropy:

```text
delta^L = A^L - Y
```

The same tidy form appears for softmax plus categorical cross-entropy:

```text
delta^L = A^L - Y
```

and for linear output plus mean squared error (with the factor of 2 folded into the
learning rate):

```text
delta^L = A^L - Y
```

In all three standard pairings, **the output error is prediction minus truth**. This
is not a coincidence; it comes from choosing the loss to match the output.

### Propagating backward

For a hidden layer `l`:

```text
delta^l = (delta^(l+1) * W^(l+1)^T) ⊙ f'(Z^l)
```

The `⊙` symbol means element-wise multiplication. In words: take the next layer's
error, pull it back through the weights, then scale by how sensitive this layer's
activation is.

The gradients for the parameters are then:

```text
dW^l = A^(l-1)^T * delta^l
db^l = sum of delta^l over rows
```

### Matching the practical code

For a network with input `X`, one hidden layer with sigmoid activation, and a sigmoid
output, binary cross-entropy, the whole backward pass is:

```text
delta2 = A2 - Y
dW2    = A1.T @ delta2
db2    = sum(delta2)
delta1 = (delta2 @ W2.T) * A1 * (1 - A1)
dW1    = X.T @ delta1
db1    = sum(delta1)
```

This is exactly the code in the practical. Nothing hidden, nothing magic: five lines
of matrix arithmetic.

### Why backpropagation is efficient

A naive numerical gradient would perturb each parameter separately and rerun the
forward pass. With `P` parameters, that costs `P` forward passes per gradient step.

Backpropagation costs roughly one forward pass plus one backward pass, regardless of
`P`. The backward pass is a small constant factor more expensive than the forward
pass. This efficiency is what makes training networks with millions of parameters
feasible.

### Vanishing and exploding gradients

The backward signal is a product of many derivatives, one per layer. If those
factors are consistently less than 1, the signal shrinks toward zero as it travels
back (**vanishing gradient**). If they are consistently greater than 1, it blows up
(**exploding gradient**).

- The sigmoid derivative is at most `0.25`, so deep sigmoid stacks vanish quickly.
- The tanh derivative is at most `1`, but still saturates near the edges.
- ReLU has derivative exactly `1` on the active side, which is why it became the
  default.
- Careful initialization, normalization layers, residual connections, and gradient
  clipping are standard defenses.

## 10. Training Neural Networks

### The loop

Training repeats a simple cycle:

1. Forward propagate a batch of rows.
2. Compute the loss.
3. Backpropagate to get the gradients.
4. Update each parameter in the opposite direction of its gradient.

```text
w <- w - learning_rate * dJ/dw
b <- b - learning_rate * dJ/db
```

### Batches, steps, and epochs

| Term | Meaning |
|------|---------|
| Batch | a small group of rows processed together |
| Step (iteration) | one parameter update |
| Epoch | one full pass over the training data |

- **Full-batch** gradient descent uses all rows per update: accurate gradients,
  slow, memory-heavy.
- **Stochastic** gradient descent uses one row: noisy but cheap.
- **Mini-batch** uses, say, 32 or 256 rows: the standard compromise.

Mini-batches add noise to the gradient, which can help escape poor regions, but the
noise also makes the loss curve jumpy. Shuffling the rows each epoch prevents the
network from learning the order.

### The learning rate

The learning rate is the single most important hyperparameter in practice.

| Learning rate | Symptom |
|---------------|---------|
| Too large | loss explodes or oscillates; parameters jump past the minimum |
| Too small | loss decreases painfully slowly; training stalls |
| Reasonable | loss decreases steadily, then flattens |

scikit-learn's `MLPClassifier` uses an adaptive solver by default (`adam`), which
maintains a per-parameter learning rate. `adam` is a strong default; plain SGD with
a well-tuned schedule can also work.

### Initialization

Weights must start **random and small**. Why not zero?

If every weight in a layer starts at the same value, every unit in that layer
computes the same output and receives the same gradient. The units stay identical
forever. This is the **symmetry breaking** problem: random initialization makes
units diverge and learn different features.

Good schemes scale the initial variance by the layer size:

- **Xavier/Glorot:** variance around `1 / fan_in` or `2 / (fan_in + fan_out)`,
  suited to sigmoid and tanh.
- **He:** variance around `2 / fan_in`, suited to ReLU.

scikit-learn handles initialization internally, but the concept matters when you
move to a framework such as PyTorch.

### Reproducibility

Neural networks are stochastic: random initialization, shuffled batches, and
hardware nondeterminism all change the result. Two runs with the same code can give
slightly different weights.

- Fix a `random_state` when you need repeatable experiments.
- Do not treat small differences as meaningful without checking run-to-run variance.
- Compare models on the same split and the same preprocessing.

## 11. Regularization and Generalization

A network with many parameters can memorize its training data. The goal is to learn
the pattern, not the noise. Several tools fight overfitting.

| Tool | What it does | Cost |
|------|--------------|------|
| More data | the best regularizer when available | expensive to collect |
| Smaller network | limits capacity | may underfit |
| L2 weight decay | penalizes large weights; scikit-learn's `alpha` | biases weights toward zero |
| Dropout | randomly disables units during training | slower training |
| Early stopping | stops when validation loss stops improving | needs a validation split |
| Data augmentation | creates varied training examples | task-specific |
| Batch normalization | stabilizes and speeds training | extra complexity |

### Weight decay

This is the same idea as Ridge from Day 06, applied to every weight matrix:

```text
J_total = J_data + alpha * sum(w^2)
```

In `MLPClassifier`, `alpha` is the L2 penalty strength. Larger `alpha` means a
simpler, flatter model.

### Dropout

During training, dropout randomly sets a fraction of activations to zero on each
batch. This prevents units from relying on specific partners and forces redundant
representations. At test time, dropout is switched off and activations are scaled to
compensate.

Dropout is most useful in large networks with lots of data. It is not a magic fix
for a tiny dataset.

### Early stopping

Train on the training set, watch the **validation** loss, and stop when it stops
improving. The version of the model with the best validation loss is the one you
keep. This is simple and effective, but it requires a validation split so the test
set stays untouched.

### Feature scaling

Standardize inputs before training. A network's gradients depend on the input
magnitudes, and unscaled features make the loss surface badly shaped. This is the
same advice as Day 06, but the effect is larger for networks because many layers
multiply the distortion.

## 12. Deep Learning and When It Is Useful

**Deep learning** means neural networks with many layers. Its defining advantage is
**representation learning**: instead of hand-designing features, the network learns
a hierarchy of features from data.

### What depth buys

- Early layers learn simple patterns: edges, frequencies, local contrasts.
- Middle layers combine them into parts.
- Later layers combine parts into objects, words, or concepts.

This hierarchy is why deep networks dominate on images, speech, and language. A
hand-engineered feature pipeline can imitate it for one task, but a deep network can
learn the right features for many tasks from raw data.

### The costs

- **Data:** deep networks are data-hungry. Millions of labeled examples are common.
- **Compute:** training and inference need GPUs or specialized hardware.
- **Tuning:** architecture, learning rate, regularization, and augmentation all
  matter.
- **Opacity:** a deep network is much harder to interpret than a coefficient table.
- **Fragility:** performance can drop under distribution shift; small adversarial
  perturbations can fool image classifiers.
- **Energy and latency:** large models are expensive to serve.

### When a neural network is the right tool

| Situation | Sensible first choice |
|-----------|-----------------------|
| Small or medium tabular data | logistic regression, linear regression, or gradient-boosted trees |
| Text, images, audio | pretrained deep networks and fine-tuning |
| Very large, complex datasets | deep networks |
| Strict interpretability | linear models or small trees |
| Tight latency or energy budget | small models, or simpler classical methods |
| Simple, roughly linear relationship | linear model first |

The honest default on tabular data is still to try a linear model and a
gradient-boosted tree. If a neural network does not beat them on held-out data, it
has not earned its complexity.

### Deep learning does not replace the fundamentals

The workflow from Day 04 still applies: define the question, split before
preprocessing, build a baseline, choose metrics from real costs, evaluate on unseen
data, and monitor after deployment. A network is just a more flexible model inside
the same honest process.

## 13. Real-World Use Case: Handwritten Digit Recognition

A postal service wants to read handwritten ZIP codes from scanned envelopes.

- **Row:** one small grayscale image, flattened into pixels.
- **Features:** pixel intensities, 64 values for an 8×8 image.
- **Target:** which digit, 0 through 9.
- **Model:** a network with one hidden layer of 64 ReLU units and a 10-way softmax
  output.
- **Loss:** categorical cross-entropy.
- **Metrics:** overall accuracy, per-class recall, and the confusion matrix. Some
  digit pairs are systematically confused, and that is what an operator needs to
  know.

Why the ideas from today matter:

- **A single neuron is not enough:** digits are not separated by one hyperplane in
  pixel space. Hidden layers build the combination features that separate them.
- **Forward propagation:** cheap and fast at prediction time, which matters for mail
  sorting.
- **Backpropagation:** lets the network learn thousands of weights from labeled
  examples.
- **Softmax and cross-entropy:** produce calibrated-looking class probabilities, so
  a low-confidence digit can be routed to a human.
- **Regularization and validation:** prevent memorizing the training envelopes and
  catch overfitting before deployment.

What could go wrong:

- **Leakage:** normalizing with statistics from the test envelopes, or reusing the
  same digits in training and test.
- **Distribution shift:** a new pen, paper, or scanner changes the pixels. Monitor
  and retrain.
- **Class imbalance:** some digits may be rare; accuracy alone can hide poor recall.
- **Overconfidence:** a wrong digit predicted with 99% confidence is dangerous; check
  calibration and route uncertain cases to humans.
- **Fairness and privacy:** handwriting varies by language and culture; audits and
  human oversight are required.

## 14. Limitations and Pitfalls

- **A single neuron is still a linear classifier.** It cannot solve XOR or any
  non-separable pattern without added features or hidden layers.
- **More layers are not automatically better.** Depth makes optimization harder and
  can overfit without enough data and regularization.
- **Neural networks are data-hungry.** On small tabular datasets, simpler models
  often win.
- **They are sensitive to preprocessing.** Scaling, missing values, and outliers all
  matter.
- **They are sensitive to hyperparameters.** Learning rate, architecture, and
  regularization can change results dramatically.
- **They are stochastic.** Two runs differ; report variance, not a single lucky
  number.
- **They are hard to interpret.** Feature importance and coefficients are not
  directly available; post-hoc explanation methods are approximations.
- **They can be miscalibrated and overconfident.** A high softmax score is not
  always a reliable probability.
- **They are vulnerable to distribution shift and adversarial inputs.** Validate on
  data that resembles deployment.
- **Universal approximation is not a promise.** It says a big enough network *can*
  represent the function, not that you can train it with the data you have.
- **Correlation is not causation.** A learned association is still an association.

## 15. Common Mistakes

1. Stacking linear layers and expecting depth to help; without a nonlinear
   activation, the stack is one linear model.
2. Forgetting to scale the inputs before training.
3. Initializing all weights to zero and breaking symmetry.
4. Using a learning rate that is far too large or far too small.
5. Watching training loss only, with no validation split.
6. Tuning the architecture and learning rate on the test set.
7. Making the network huge without regularization, or tiny and permanently
   underfitting.
8. Using accuracy on an imbalanced classification problem.
9. Expecting a single perceptron to solve XOR or another non-separable problem.
10. Forgetting to set a random seed, then over-reading small run-to-run differences.
11. Comparing models trained on different splits or different preprocessing.
12. Reaching for a deep network on a small tabular dataset before trying a linear
    model or a tree.
13. Mismatching the output activation and the loss (softmax with squared error,
    sigmoid with categorical targets).
14. Treating the universal approximation theorem as a guarantee of generalization.
15. Assuming a deeper network always generalizes better than a shallower one.

## 16. Putting It Together

Before trusting a neural network, ask:

1. Is the target regression, binary, multiclass, or multilabel, and does the output
   activation and loss match?
2. Did I split the data before fitting any scaler or imputer?
3. Did I standardize the features?
4. Did I compare against a simple baseline, such as logistic regression, a linear
   model, or a gradient-boosted tree?
5. Is the architecture justified, or is it copied without thought?
6. Did training converge, and does the loss curve look healthy?
7. Is there a validation split, and did I use it for early stopping and tuning?
8. Did I check overfitting by comparing training and validation metrics?
9. Did I set a random seed and check that results are stable across a few runs?
10. Did I report metrics beyond accuracy, especially per-class recall and a
    confusion matrix?
11. Have I checked for data leakage and distribution shift?
12. Could a simpler, more interpretable model do the job just as well?

If the answers are reasonable, and the network beats the baseline on held-out data,
it has earned its complexity.

## 17. Summary

```text
neuron:      z = w·x + b,  a = f(z)
perceptron:  y_hat = step(z),  w <- w + lr*(y - y_hat)*x
network:     input -> hidden layers (nonlinear) -> output
training:    forward -> loss -> backprop -> update
```

- A neuron is a weighted sum plus a bias, passed through an activation.
- The activation provides the nonlinearity; without it, depth is an illusion.
- The perceptron uses a step activation and a mistake-driven update, and it
  converges only on linearly separable data.
- XOR is not linearly separable; no single perceptron can solve it.
- A hidden layer creates new features, so a two-layer network can solve XOR.
- Layers stack into a feedforward network; depth composes features hierarchically.
- Forward propagation computes predictions; backpropagation assigns blame using the
  chain rule.
- For standard output and loss pairings, the output error is `prediction - truth`.
- Scaling, initialization, learning rate, and regularization determine whether
  training succeeds.
- Neural networks are powerful for large, complex, high-dimensional data, but they
  are not automatically better than simpler models on small tabular problems.
- The honest ML workflow still applies: split first, baseline, validate, and
  evaluate on unseen data.

## Outcome

Learners can explain how an artificial neuron computes its output, describe the
major activation functions and why nonlinearity matters, apply the perceptron
learning rule, prove informally why XOR defeats a linear classifier, explain how a
hidden layer solves it, trace forward propagation, describe backpropagation as the
chain rule applied layer by layer, and recognize when a neural network is the right
tool and when a simpler model is better.
