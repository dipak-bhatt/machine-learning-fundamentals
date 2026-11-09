# Day 08 — Practical: Perceptron and Neural Networks

**Module:** Module 2: Supervised Learning

In this practical we build the bridge from a single neuron to a small neural
network. We will:

- Compare the step, sigmoid, tanh, and ReLU activations on the same input.
- Compute one neuron's output by hand.
- Implement the perceptron learning rule in NumPy.
- Train it on linearly separable data and watch it converge.
- Watch it fail on XOR.
- Solve XOR with a hand-written hidden layer and then with scikit-learn.
- Train networks on the two-moons data and on handwritten digits.
- Read a training curve, vary the architecture, and see overfitting on a tiny
  dataset.

Everything is generated or bundled with scikit-learn, so the notebook can live in
this Day 08 folder without any path configuration.

## 1. Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    confusion_matrix,
    classification_report,
)

RANDOM_STATE = 42
```

`RANDOM_STATE` makes the splits, generated data, and network initializations
repeatable. Neural networks are stochastic, so fixing the seed is the only way to
compare runs fairly.

## 2. Warm-Up: Activations and One Neuron

An artificial neuron computes a weighted sum and passes it through an activation.
Let us line up four common activations for the same input.

```python
def step(z):
    return np.where(z >= 0, 1, 0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def tanh(z):
    return np.tanh(z)


def relu(z):
    return np.maximum(0, z)


for z in [-2, 0, 2]:
    print(z, int(step(z)), round(float(sigmoid(z)), 3), round(float(tanh(z)), 3), int(relu(z)))
```

Expected output:

```text
-2 0 0.119 -0.964 0
0 1 0.5 0.0 0
2 1 0.881 0.964 2
```

At `z = 0` the step turns on, the sigmoid says "50/50", and tanh is exactly zero.
Negative inputs are switched off by ReLU. The activations encode different ideas:
a hard decision, a probability, a signed signal, or a non-negative one.

Now compute one neuron by hand. Take `x = [1, 2]`, `w = [0.5, -0.5]`, and `b = 0`.

```python
x = np.array([1.0, 2.0])
w = np.array([0.5, -0.5])
b = 0.0

z = float(np.dot(w, x) + b)
print("weighted sum z:", z)
print("step:", int(step(z)))
print("sigmoid:", round(float(sigmoid(z)), 3))
```

Expected output:

```text
weighted sum z: -0.5
step: 0
sigmoid: 0.378
```

The weighted sum is negative, so the step says 0 and the sigmoid says the
probability is below 0.5. This is exactly logistic regression when the activation
is the sigmoid, and exactly a perceptron when it is the step.

## 3. Implement a Perceptron

The perceptron is the simplest trainable neuron. It predicts with a step function
and updates its weights only when it makes a mistake.

```python
class Perceptron:
    def __init__(self, learning_rate=0.1, n_epochs=100):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.errors_ = []
        for _ in range(self.n_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                y_hat = 1 if np.dot(self.w, xi) + self.b >= 0 else 0
                update = self.learning_rate * (yi - y_hat)
                self.w += update * xi
                self.b += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            if errors == 0:
                break
        return self

    def predict(self, X):
        return (X @ self.w + self.b >= 0).astype(int)
```

The update is `w <- w + learning_rate * (y - y_hat) * x`. When the prediction is
correct, `y - y_hat` is zero and nothing changes. When it is wrong, the weights move
toward the correct side. `errors_` records how many mistakes were made in each
epoch.

## 4. Train on Linearly Separable Data

Generate two well-separated blobs and split them.

```python
Xb, yb = make_blobs(
    n_samples=200, centers=2, cluster_std=1.2, random_state=RANDOM_STATE
)
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    Xb, yb, test_size=0.25, random_state=RANDOM_STATE, stratify=yb
)

perceptron = Perceptron(learning_rate=0.1, n_epochs=100).fit(Xb_train, yb_train)

print("Train accuracy:", round(accuracy_score(yb_train, perceptron.predict(Xb_train)), 3))
print("Test accuracy:", round(accuracy_score(yb_test, perceptron.predict(Xb_test)), 3))
print("Epochs run:", len(perceptron.errors_))
print("Errors per epoch:", perceptron.errors_)
print("Weights:", np.round(perceptron.w, 3), "bias:", round(perceptron.b, 3))
```

Expected output:

```text
Train accuracy: 1.0
Test accuracy: 1.0
Epochs run: 2
Errors per epoch: [3, 0]
Weights: [ 0.83 -0.53] bias: 0.1
```

The perceptron made three mistakes in the first pass and none in the second, so it
stopped. This is the convergence theorem in action: on linearly separable data, the
perceptron finishes in finitely many updates. The exact weights depend on the starting
point and the row order, but any separating line is accepted.

Try a few learning rates. A larger rate makes bigger jumps; a smaller rate takes
more epochs. On separable data the model still converges, because separability does
not depend on the size of the weights.

## 5. The Perceptron Fails on XOR

XOR is the smallest non-separable problem. Its four points sit on opposite corners
of a square, so no single line can separate the classes.

```python
Xxor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
yxor = np.array([0, 1, 1, 0])

xor_perceptron = Perceptron(learning_rate=0.1, n_epochs=100).fit(Xxor, yxor)

print("Predictions:", xor_perceptron.predict(Xxor))
print("Accuracy:", accuracy_score(yxor, xor_perceptron.predict(Xxor)))
print("Errors per epoch:", xor_perceptron.errors_[:6], "...", xor_perceptron.errors_[-1])
```

Expected output:

```text
Predictions: [1 1 0 0]
Accuracy: 0.5
Errors per epoch: [3, 3, 4, 4, 4, 4] ... 4
```

The perceptron never reaches zero errors. It predicts two of the four points
correctly and keeps cycling, because no line can do better than 50% here. This is
not a bug in the code; it is a limitation of the model. Section 5 of the theory file
proves that no choice of `w` and `b` can satisfy all four XOR rows.

Contrast this with the blobs in Section 4, where the error count reached zero after
two epochs. The difference is **linear separability**, not the learning rate or the
number of epochs.

## 6. Solve XOR with a Hidden Layer

A hidden layer creates new features. We build a tiny network by hand:

```text
2 inputs -> 4 hidden units (sigmoid) -> 1 output unit (sigmoid)
```

We train it with full-batch gradient descent and binary cross-entropy.

```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

rng = np.random.default_rng(42)
n_input, n_hidden, n_output = 2, 4, 1

W1 = rng.normal(0, 1, size=(n_input, n_hidden))
b1 = np.zeros((1, n_hidden))
W2 = rng.normal(0, 1, size=(n_hidden, n_output))
b2 = np.zeros((1, n_output))


def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2


def bce(y, p):
    eps = 1e-9
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


learning_rate = 0.5
n_epochs = 5000

for epoch in range(n_epochs):
    z1, a1, z2, a2 = forward(X, W1, b1, W2, b2)

    delta2 = (a2 - y) / len(X)
    dW2 = a1.T @ delta2
    db2 = delta2.sum(axis=0, keepdims=True)

    delta1 = (delta2 @ W2.T) * a1 * (1 - a1)
    dW1 = X.T @ delta1
    db1 = delta1.sum(axis=0, keepdims=True)

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch in [0, 999, 4999]:
        print(f"epoch {epoch:4d}  loss {bce(y, a2):.4f}  probabilities {np.round(a2.ravel(), 3)}")

print("Final probabilities:", np.round(a2.ravel(), 3))
print("Predicted labels:", (a2.ravel() >= 0.5).astype(int))
print("Accuracy:", round(float(np.mean((a2.ravel() >= 0.5) == y.ravel())), 3))
```

Expected output:

```text
epoch    0  loss 0.7568  probabilities [0.597 0.648 0.716 0.741]
epoch  999  loss 0.0645  probabilities [0.031 0.931 0.942 0.092]
epoch 4999  loss 0.0028  probabilities [0.001 0.997 0.998 0.004]
Final probabilities: [0.001 0.997 0.998 0.004]
Predicted labels: [0 1 1 0]
Accuracy: 1.0
```

The hidden layer turned a non-separable problem into a separable one. The backward
pass is the same five-line pattern as the theory:

```text
delta2 = prediction - truth
dW2    = hidden_activations.T @ delta2
delta1 = (delta2 @ W2.T) * a1 * (1 - a1)
dW1    = inputs.T @ delta1
```

A few details worth noticing:

- Weights start **random**. If they all started at zero, the four hidden units would
  be identical and stay identical.
- The hidden activation is the sigmoid. Its derivative is `a1 * (1 - a1)`, which
  appears in `delta1`.
- The learning rate is large (`0.5`) because this is a tiny, full-batch problem.
- The loss drops from `0.757` to `0.003` and the four points become perfectly
  classified.

## 7. Solve XOR with scikit-learn

The same network is one line with `MLPClassifier`. We use `tanh` hidden units and the
`lbfgs` solver, which handles this tiny deterministic problem well.

```python
xor_mlp = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation="tanh",
    solver="lbfgs",
    max_iter=5000,
    random_state=RANDOM_STATE,
).fit(Xxor, yxor)

print("Predictions:", xor_mlp.predict(Xxor))
print("Probabilities:", np.round(xor_mlp.predict_proba(Xxor), 3))
print("Final loss:", round(float(xor_mlp.loss_), 4))
```

Expected output:

```text
Predictions: [0 1 1 0]
Probabilities: [[1. 0.]
 [0. 1.]
 [0. 1.]
 [1. 0.]]
Final loss: 0.0031
```

The network solves XOR just like our hand-written version. The architecture and
optimizer are choices: on this seed, the default `relu` with `adam` can stall at
50% accuracy, while `tanh` with `lbfgs` succeeds. That is a real property of neural
network training, not a mistake. Section 14 asks you to compare the two.

## 8. Nonlinear Data: Two Moons

XOR is a toy. The two-moons dataset is a more realistic curved boundary. We compare
logistic regression with a small neural network on the same split.

```python
Xm, ym = make_moons(n_samples=400, noise=0.25, random_state=RANDOM_STATE)
Xm_train, Xm_test, ym_train, ym_test = train_test_split(
    Xm, ym, test_size=0.25, random_state=RANDOM_STATE, stratify=ym
)

moons_logistic = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
).fit(Xm_train, ym_train)

moons_mlp = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, random_state=RANDOM_STATE),
).fit(Xm_train, ym_train)

for name, model in [("Logistic regression", moons_logistic), ("Neural network", moons_mlp)]:
    proba = model.predict_proba(Xm_test)[:, 1]
    accuracy = accuracy_score(ym_test, model.predict(Xm_test))
    loss = log_loss(ym_test, proba)
    print(f"{name:20s} accuracy={accuracy:.3f}  log_loss={loss:.3f}")
```

Expected output:

```text
Logistic regression  accuracy=0.920  log_loss=0.230
Neural network       accuracy=0.960  log_loss=0.158
```

The linear baseline is already strong at 92%, but the network gains four accuracy
points and a much better log loss. It can bend its boundary around the two crescents;
logistic regression cannot.

Standardization is not optional here. The pipeline fits the scaler on the training
rows only, so the test set stays unseen. This is the same leakage rule as Day 06 and
Day 07.

## 9. Visualize the Decision Surface

A picture makes the difference clear. We evaluate the network on a fine grid and
color each point by its predicted probability.

```python
xx, yy = np.meshgrid(
    np.linspace(Xm[:, 0].min() - 0.5, Xm[:, 0].max() + 0.5, 300),
    np.linspace(Xm[:, 1].min() - 0.5, Xm[:, 1].max() + 0.5, 300),
)
grid = np.c_[xx.ravel(), yy.ravel()]
zz = moons_mlp.predict_proba(grid)[:, 1].reshape(xx.shape)

plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.7)
plt.colorbar(label="P(class 1)")
plt.scatter(Xm_test[:, 0], Xm_test[:, 1], c=ym_test, cmap="RdBu_r", edgecolor="k")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Neural network decision surface (two moons)")
plt.show()
```

The boundary is curved, unlike the straight line logistic regression would draw. The
hidden units combine the two inputs into intermediate features, and the output unit
combines those features into a flexible boundary.

## 10. The Training Curve and Capacity

`MLPClassifier` stores the loss after each iteration in `loss_curve_`. A healthy
curve decreases and then flattens.

```python
losses = moons_mlp.named_steps["mlpclassifier"].loss_curve_
print("Iterations:", len(losses))
print("First 3 losses:", [round(float(v), 3) for v in losses[:3]])
print("Last 3 losses:", [round(float(v), 3) for v in losses[-3:]])

plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.title("MLP training loss on the two-moons data")
plt.show()
```

Expected output:

```text
Iterations: 1219
First 3 losses: [0.689, 0.684, 0.679]
Last 3 losses: [0.167, 0.167, 0.167]
```

The loss starts near `0.69`, the value of an uninformed coin flip, and settles near
`0.17`. The long flat stretch at the end is the optimizer making tiny refinements.

Now vary the size of the hidden layer:

```python
for hidden in [(4,), (16,), (64,)]:
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=hidden, max_iter=2000, random_state=RANDOM_STATE),
    ).fit(Xm_train, ym_train)

    proba = model.predict_proba(Xm_test)[:, 1]
    accuracy = accuracy_score(ym_test, model.predict(Xm_test))
    loss = log_loss(ym_test, proba)
    print(f"hidden={str(hidden):6s} accuracy={accuracy:.3f}  log_loss={loss:.3f}")
```

Expected output:

```text
hidden=(4,)   accuracy=0.900  log_loss=0.256
hidden=(16,)  accuracy=0.960  log_loss=0.158
hidden=(64,)  accuracy=0.930  log_loss=0.179
```

Four hidden units underfit: the boundary is almost linear. Sixteen units are enough
to bend around the moons. Sixty-four units are more than enough and slightly worse,
a hint of overfitting on only 300 training rows. **More capacity is not free.**

## 11. Multiclass Classification: Handwritten Digits

The digits dataset contains 1,797 small images of handwritten digits. Each row is an
8×8 image flattened into 64 pixel values, and the target is the digit 0 through 9.

```python
digits = load_digits()
Xd = digits.data
yd = digits.target

Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    Xd, yd, test_size=0.25, random_state=RANDOM_STATE, stratify=yd
)

print("Data shape:", Xd.shape)
print("Classes:", np.unique(yd))
```

Expected output:

```text
Data shape: (1797, 64)
Classes: [0 1 2 3 4 5 6 7 8 9]
```

Train a network with one hidden layer of 64 ReLU units and a 10-way softmax output:

```python
digits_mlp = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000, random_state=RANDOM_STATE),
).fit(Xd_train, yd_train)

digits_pred = digits_mlp.predict(Xd_test)

print("Train accuracy:", round(accuracy_score(yd_train, digits_mlp.predict(Xd_train)), 3))
print("Test accuracy:", round(accuracy_score(yd_test, digits_pred), 3))
print("Test log loss:", round(log_loss(yd_test, digits_mlp.predict_proba(Xd_test)), 3))
print(confusion_matrix(yd_test, digits_pred))
print(classification_report(yd_test, digits_pred, digits=3))
```

Expected output:

```text
Train accuracy: 1.0
Test accuracy: 0.98
Test log loss: 0.098
[[44  0  0  0  1  0  0  0  0  0]
 [ 0 45  1  0  0  0  0  0  0  0]
 [ 0  0 44  0  0  0  0  0  0  0]
 [ 0  0  0 46  0  0  0  0  0  0]
 [ 0  0  0  0 44  0  0  1  0  0]
 [ 0  0  0  0  0 44  0  0  0  2]
 [ 0  0  0  0  0  0 45  0  0  0]
 [ 0  0  0  0  0  0  0 45  0  0]
 [ 1  2  1  0  0  0  0  0 39  0]
 [ 0  0  0  0  0  0  0  0  0 45]]
              precision    recall  f1-score   support

           0      0.978     0.978     0.978        45
           1      0.957     0.978     0.968        46
           2      0.957     1.000     0.978        44
           3      1.000     1.000     1.000        46
           4      0.978     0.978     0.978        45
           5      1.000     0.957     0.978        46
           6      1.000     1.000     1.000        45
           7      0.978     1.000     0.989        45
           8      1.000     0.907     0.951        43
           9      0.957     1.000     0.978        45

    accuracy                          0.980       450
   macro avg      0.981     0.980     0.980       450
weighted avg      0.981     0.980     0.980       450
```

The model reaches 98% test accuracy. Training accuracy is 1.0, which means it fully
memorized the training rows; the small train/test gap is still acceptable here.
Digit 8 is the weakest class: four of its images are predicted as 0, 1, or 2. Its
recall is 0.907, and that is the kind of detail a single accuracy number would hide.

Now compare with logistic regression on exactly the same split:

```python
digits_logistic = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
).fit(Xd_train, yd_train)

print("Logistic regression test accuracy:", round(accuracy_score(yd_test, digits_logistic.predict(Xd_test)), 3))
```

Expected output:

```text
Logistic regression test accuracy: 0.978
```

This is an important, honest result: the neural network scores 0.980 and logistic
regression scores 0.978. The extra complexity buys almost nothing on this small,
clean dataset. Always compare against a simpler baseline before celebrating a neural
network.

## 12. Inspect the Architecture

scikit-learn stores the learned weight matrices and biases in `coefs_` and
`intercepts_`.

```python
mlp = digits_mlp.named_steps["mlpclassifier"]

print("Weight shapes:", [w.shape for w in mlp.coefs_])
print("Bias shapes:", [b.shape for b in mlp.intercepts_])
print("Hidden activation:", mlp.activation)
print("Output activation:", mlp.out_activation_)
print("Total parameters:", sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_))
```

Expected output:

```text
Weight shapes: [(64, 64), (64, 10)]
Bias shapes: [(64,), (10,)]
Hidden activation: relu
Output activation: softmax
Total parameters: 4810
```

Read the shapes carefully:

- `(64, 64)` connects the 64 inputs to the 64 hidden units. That is `64 * 64 = 4096`
  weights plus 64 biases.
- `(64, 10)` connects the 64 hidden units to the 10 output classes. That is `640`
  weights plus 10 biases.
- Total: `4096 + 64 + 640 + 10 = 4810` parameters.

The hidden activation is ReLU and the output activation is softmax, which matches
the categorical cross-entropy loss. A softmax output chooses the digit with the
largest probability.

## 13. Overfitting on a Tiny Dataset

A large network can memorize a small training set. Train a 256-unit network on only
100 digit images and evaluate it on the full test set.

```python
tiny_X = Xd_train[:100]
tiny_y = yd_train[:100]

tiny_mlp = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(256,), max_iter=2000, random_state=RANDOM_STATE),
).fit(tiny_X, tiny_y)

print("Tiny-data train accuracy:", round(accuracy_score(tiny_y, tiny_mlp.predict(tiny_X)), 3))
print("Tiny-data test accuracy:", round(accuracy_score(yd_test, tiny_mlp.predict(Xd_test)), 3))
```

Expected output:

```text
Tiny-data train accuracy: 1.0
Tiny-data test accuracy: 0.82
```

Perfect training accuracy and 82% test accuracy: the network memorized 100 images
instead of learning the digit patterns. More units made it worse, not better. The
fixes are the ones from Section 11 of the theory file: more data, a smaller network,
weight decay (`alpha`), or early stopping on a validation split.

## 14. Try It Yourself

Use the variables from this notebook.

1. Train the perceptron on the two-moons data with `Perceptron(n_epochs=200)`. What
   accuracy does it reach, and why can it never reach 100%?
2. Vary the perceptron learning rate `[0.01, 0.1, 1.0]` on the blobs. How does the
   number of epochs change? Does the final accuracy change?
3. Fit `MLPClassifier` on XOR with the default `activation="relu"` and `solver="adam"`.
   Does it solve the problem on this seed? Try `random_state` values `0` and `1`.
4. Add the interaction feature `x1 * x2` to the XOR input and train the perceptron on
   the three features. Does it now solve XOR? This is the hand-engineered version of
   what a hidden layer learns.
5. Vary `hidden_layer_sizes` on the moons data through `(2,)`, `(8,)`, `(32,)`, and
   `(8, 8)`. Plot or print test accuracy and log loss. Where does capacity stop
   helping?
6. Vary `alpha` from `0.0001` to `1.0` on the moons network. What happens to the
   training loss, the test loss, and the decision boundary as the penalty grows?
7. Compare `activation="logistic"`, `"tanh"`, and `"relu"` on the same moons split.
   Which converges fastest? Which generalizes best?
8. Train the digits network on only 20% of the training rows. Compare it with
   logistic regression on the same reduced data. Does the neural network still have
   an edge?
9. Optional: rebuild the XOR network in PyTorch. The forward pass, loss, and
   backpropagation map directly to the NumPy code in Section 6.

Hint for question 4:

```python
Xxor_interaction = np.c_[Xxor, Xxor[:, 0] * Xxor[:, 1]]
interaction_perceptron = Perceptron(learning_rate=0.1, n_epochs=100).fit(
    Xxor_interaction, yxor
)
print("Predictions:", interaction_perceptron.predict(Xxor_interaction))
print("Accuracy:", accuracy_score(yxor, interaction_perceptron.predict(Xxor_interaction)))
```

Hint for question 3:

```python
for seed in [0, 1, 42]:
    model = MLPClassifier(
        hidden_layer_sizes=(4,), max_iter=5000, random_state=seed
    ).fit(Xxor, yxor)
    print(seed, model.predict(Xxor), round(accuracy_score(yxor, model.predict(Xxor)), 3))
```

The lesson is not that one seed is "right". It is that initialization and optimizer
choices can decide whether a small network escapes a bad region.

## 15. What We Learned

- A neuron is a weighted sum plus a bias, passed through an activation.
- Step, sigmoid, tanh, and ReLU encode different behaviors; the activation is what
  makes depth meaningful.
- The perceptron updates only on mistakes and converges only on linearly separable
  data.
- XOR is not linearly separable, so a single perceptron cannot solve it; we saw the
  error count never reach zero.
- A hidden layer creates new features and turns XOR into a solvable problem.
- Forward propagation is repeated linear-plus-activation; backpropagation is the
  chain rule applied backward, and for standard losses the output error is
  `prediction - truth`.
- Scaling, random initialization, learning rate, and regularization all matter.
- The two-moons network beat logistic regression, 0.960 to 0.920.
- On the digits data, the network beat logistic regression only narrowly, 0.980 to
  0.978. Complexity must be earned on held-out data.
- A large network on a tiny dataset overfits: 1.0 training accuracy, 0.82 test
  accuracy.
- Always compare against a baseline, report more than accuracy, and keep the test set
  untouched until the final check.

## Outcome

Learners can compute a neuron's output, explain the role of activations, implement and
train a perceptron, explain why XOR defeats a linear classifier, build a hidden layer
that solves it, trace forward and backward passes, train a small scikit-learn network
on nonlinear and multiclass data, read a training curve, and recognize overfitting and
the limits of extra capacity.
