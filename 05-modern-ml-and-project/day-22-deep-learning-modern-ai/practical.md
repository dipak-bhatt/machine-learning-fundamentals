# Day 22 — Practical: A Small Neural Network

**Module:** Module 6: Modern ML and Project

In this practical we will:

- Load the built-in digits dataset, a small image-classification problem.
- Compare logistic regression and a random forest with cross-validation.
- Train a small multilayer perceptron (MLP) and observe underfitting from early
  stopping on a small dataset.
- Train the same network without early stopping and compare again.
- Plot the training loss curve.
- Evaluate the selected network once on the test set.
- Inspect the confusion matrix and a misclassified image.
- Discuss cost, data requirements, and limitations.

We use scikit-learn's `MLPClassifier`, which implements the same ideas as a PyTorch or
TensorFlow network without the extra dependency. A PyTorch sketch is included at the
end for reference.

## 1. Setup

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

RANDOM_STATE = 42
```

## 2. Load the Digits Dataset

```python
digits = load_digits()
X, y = digits.data, digits.target

print("data shape:", X.shape)
print("image shape:", digits.images.shape)
print("classes:", np.unique(y))
```

Expected output:

```text
data shape: (1797, 64)
image shape: (1797, 8, 8)
classes: [0 1 2 3 4 5 6 7 8 9]
```

Each image is 8x8 pixels, flattened into 64 features. There are 1,797 examples across
10 digit classes. This is tiny by deep-learning standards, which is exactly why it is
a good place to start.

Plot a few examples:

```python
fig, axes = plt.subplots(1, 8, figsize=(12, 2))

for ax, image, label in zip(axes, digits.images, digits.target):
    ax.imshow(image, cmap="gray")
    ax.set_title(str(label))
    ax.axis("off")

plt.show()
```

## 3. Split and Scale

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

print("train:", X_train.shape, "test:", X_test.shape)
```

Expected output:

```text
train: (1347, 64) test: (450, 64)
```

Neural networks are sensitive to feature scale, so standardize inside the pipeline.

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
```

## 4. Classical Baselines

```python
logistic = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000),
)

forest = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
)

for name, model in [("logistic", logistic), ("random forest", forest)]:
    scores = cross_val_score(model, X_train, y_train, cv=cv)
    print(f"{name:14s} CV accuracy={scores.mean():.4f} std={scores.std():.4f}")
```

Expected output:

```text
logistic       CV accuracy=0.9658 std=0.0104
random forest  CV accuracy=0.9770 std=0.0109
```

Both are strong. Any neural network has to beat these to justify its cost.

## 5. A First MLP with Early Stopping

```python
mlp_early = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(64,),
        max_iter=500,
        early_stopping=True,
        random_state=RANDOM_STATE,
    ),
)

scores = cross_val_score(mlp_early, X_train, y_train, cv=cv)
print("CV accuracy:", round(scores.mean(), 4))
print("std:", round(scores.std(), 4))
```

Expected output:

```text
CV accuracy: 0.9465
std: 0.0148
```

The network underperforms both classical models. Early stopping holds out 10% of each
fold for validation; with only about 1,078 training rows per fold, that is roughly 108
rows, which is a noisy signal. It stops after a few dozen epochs, before the network
has learned enough. This is a realistic failure mode, not a bug.

## 6. Turn Off Early Stopping

Train the same architecture for a fixed maximum number of epochs.

```python
mlp = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(64,),
        max_iter=1000,
        random_state=RANDOM_STATE,
    ),
)

scores = cross_val_score(mlp, X_train, y_train, cv=cv)
print("CV accuracy:", round(scores.mean(), 4))
print("std:", round(scores.std(), 4))
```

Expected output:

```text
CV accuracy: 0.9770
std: 0.0089
```

Now the network matches the random forest and beats logistic regression, with lower
fold-to-fold variance. The only change was training long enough.

## 7. A Larger Network

More capacity does not automatically help.

```python
mlp_big = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=1000,
        random_state=RANDOM_STATE,
    ),
)

scores = cross_val_score(mlp_big, X_train, y_train, cv=cv)
print("CV accuracy:", round(scores.mean(), 4))
print("std:", round(scores.std(), 4))
```

Expected output:

```text
CV accuracy: 0.9733
std: 0.0079
```

The larger network is slightly worse on cross-validation and slower to train. With
only 1,347 training examples, the extra parameters do not pay for themselves.

## 8. Training Loss Curve

Fit the selected architecture on the full training set and plot the loss.

```python
mlp.fit(X_train, y_train)
network = mlp.named_steps["mlpclassifier"]

plt.plot(network.loss_curve_)
plt.xlabel("iteration")
plt.ylabel("training loss")
plt.title("MLP training loss")
plt.show()

print("iterations:", network.n_iter_)
print("first loss:", round(network.loss_curve_[0], 4))
print("last loss:", round(network.loss_curve_[-1], 4))
```

Expected output:

```text
iterations: 164
first loss: 2.3137
last loss: 0.0062
```

The loss falls quickly at first and then flattens. A curve that keeps falling while
validation loss rises would signal overfitting. Here the training loss is very low by
the end, so the validation comparison is what tells us the model generalizes.

## 9. Evaluate Once on the Test Set

```python
predictions = mlp.predict(X_test)

print("test accuracy:", round(accuracy_score(y_test, predictions), 4))
```

Expected output:

```text
test accuracy: 0.98
```

Compare with the logistic baseline, evaluated once on the same test set:

```python
logistic.fit(X_train, y_train)
baseline_predictions = logistic.predict(X_test)

print("logistic test accuracy:", round(accuracy_score(y_test, baseline_predictions), 4))
```

Expected output:

```text
logistic test accuracy: 0.9778
```

The neural network's edge is 0.22 percentage points: one test example out of 450. On
this dataset, that difference is not meaningful. The honest conclusion is that the
network matches a well-tuned linear model at higher cost.

## 10. Confusion Matrix and Errors

```python
cm = confusion_matrix(y_test, predictions)
print(cm)
```

Expected output:

```text
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
```

```python
print(classification_report(y_test, predictions, digits=4))
```

Expected output (abridged):

```text
              precision    recall  f1-score   support

           0     0.9778    0.9778    0.9778        45
           1     0.9574    0.9783    0.9677        46
           2     0.9565    1.0000    0.9778        44
           3     1.0000    1.0000    1.0000        46
           4     0.9778    0.9778    0.9778        45
           5     1.0000    0.9565    0.9778        46
           6     1.0000    1.0000    1.0000        45
           7     0.9783    1.0000    0.9890        45
           8     1.0000    0.9070    0.9512        43
           9     0.9574    1.0000    0.9783        45

    accuracy                         0.9800       450
```

The weakest class is 8, with recall 0.907. Four of its 43 examples are confused with
other digits. Visualize one misclassification:

```python
errors = np.where(predictions != y_test)[0]
print("number of errors:", len(errors))

first = errors[0]
print("true:", y_test[first], "predicted:", predictions[first])

test_images = X_test.reshape(-1, 8, 8)
plt.imshow(test_images[first], cmap="gray")
plt.title(f"true {y_test[first]}, predicted {predictions[first]}")
plt.axis("off")
plt.show()
```

Expected output:

```text
number of errors: 9
true: 8 predicted: 1
```

An 8 that looks like a 1 is a plausible human mistake. Inspecting errors often teaches
more than the aggregate score.

## 11. Reference: The Same Network in PyTorch

The course environment may not have `torch` installed. If it does, the equivalent
model is:

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

Training loops over batches, computes the loss, calls `loss.backward()`, and steps the
optimizer. The concepts are identical to the scikit-learn version: forward pass, loss,
backward pass, update.

A pretrained transformer is used through a high-level interface rather than by
training from scratch:

```python
# pip install transformers
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("This course made the workflow clear."))
```

This is optional and requires network access and a large download. The point is that
most practical LLM use is adaptation and prompting of a pretrained model, not training
a language model from zero.

## 12. Cost, Data, and When to Use Deep Learning

On this dataset:

- Logistic regression trains in under a second.
- The random forest takes a few seconds.
- The MLP takes longer and adds a dependency and many hyperparameters.
- All three land within about one percentage point.

The decision is not close. For small tabular data, start classical. Reach for a neural
network when:

- The inputs are raw and high-dimensional: images, audio, video, long text.
- You have a large labeled dataset or a pretrained model to fine-tune.
- The pattern is genuinely nonlinear and classical models plateau.
- You can afford GPU compute, latency, and maintenance.

Deep learning is a tool with a cost curve. Use it when the task needs it.

## 13. Try It Yourself

1. **Tune the hidden size.** Try `(16,)`, `(32,)`, `(64,)`, `(128,)` and plot CV
   accuracy. Where does it plateau?
2. **Change the activation.** Compare `relu` and `tanh`.
3. **Change the learning rate.** Try `0.001`, `0.01`, and `0.1`. Which converges
   fastest, and which becomes unstable?
4. **Add regularization.** Try `alpha=0.01` and `alpha=0.1`. Does CV improve?
5. **Use early stopping properly.** Increase the validation fraction with
   `validation_fraction=0.2` and `n_iter_no_change=20`.
6. **Augment the images.** Shift each training image by one pixel and retrain. Does
   accuracy improve?
7. **Try PCA first.** Reduce the 64 pixels to 20 components, then train the MLP. How
   does the cost and accuracy change?

## 14. What We Learned

- A multilayer perceptron stacks nonlinear layers trained by gradient descent.
- scikit-learn's `MLPClassifier` demonstrates the core mechanics without PyTorch.
- Early stopping can underfit when the validation split is small.
- Training long enough mattered more than adding capacity here.
- The MLP matched, but did not meaningfully beat, logistic regression and random
  forest on this small dataset.
- The training loss curve shows optimization; validation performance shows
  generalization.
- Confusion matrices and error inspection reveal which classes are hard.
- Deep learning is not automatically better; data size and input type decide.
- PyTorch and transformer snippets map the same concepts to production tooling.

## Outcome

Learners can train a small neural network, explain the forward pass, loss, and
backpropagation, recognize underfitting from early stopping, compare a network against
classical baselines, read a training curve and confusion matrix, and judge when deep
learning is worth its data, compute, and maintenance costs.
