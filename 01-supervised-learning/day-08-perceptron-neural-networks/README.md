# Day 08 — Perceptron and Neural Network Fundamentals

**Module:** Module 2: Supervised Learning

Every model so far has been linear in its inputs, or linear in features we built
by hand. A **neural network** is different: it stacks many simple units, each one a
tiny linear model plus a nonlinearity, and lets the combination learn patterns no
single line can capture. The story starts with the **perceptron**, the first
trainable neuron, and its famous failure on XOR.

Today we build a neuron from scratch, watch a single perceptron fail, add a hidden
layer, and see the same fix work with scikit-learn's `MLPClassifier`.

## What You Will Learn

- What an artificial neuron is: weighted inputs, a bias, and an activation.
- How the step, sigmoid, tanh, and ReLU activations differ, and why nonlinearity is
  essential once units are stacked.
- The perceptron learning rule and when it is guaranteed to converge.
- What linear separability means, and why a single perceptron cannot solve XOR.
- How input, hidden, and output layers turn a network into a flexible function.
- How forward propagation computes predictions layer by layer.
- Loss functions for regression, binary classification, and multiclass problems.
- How backpropagation uses the chain rule to assign blame to every weight.
- Why scaling, initialization, learning rate, and regularization matter more here.
- When a neural network is worth its cost, and when a simpler model is better.

## Detailed Content

- [Theory](theory.md) — the artificial neuron, activation functions, the perceptron
  learning rule, linear separability and XOR, layers and parameter counts, forward
  propagation, losses, backpropagation, training and initialization, regularization,
  deep learning, when to use neural networks, and common mistakes.
- [Practical](practical.md) — activate a single neuron by hand, implement a
  perceptron in NumPy, train it on linearly separable data, watch it fail on XOR,
  solve XOR with a hand-written hidden layer, then train scikit-learn networks on
  two-moons and handwritten digits.

## Files

No CSV is needed. Every dataset is generated or bundled with scikit-learn:

- **Two blobs** (`make_blobs`) — a linearly separable binary problem for the
  perceptron.
- **XOR** — four points that no single line can separate.
- **Two moons** (`make_moons`) — a curved binary problem where a hidden layer helps.
- **Digits** (`load_digits`) — 1,797 handwritten 8×8 images across ten classes.

The notebook should be created inside this Day 08 folder. Nothing is downloaded, so
there is no path to configure.

## Before You Start

- Day 07: the sigmoid, log loss, decision boundaries, and classification metrics.
- Day 06: gradient descent, learning rate, feature scaling, and regularization.
- Day 04: train/test splits, baselines, overfitting, and honest evaluation.

## Outcome

Learners can explain how a single neuron computes its output, apply the perceptron
learning rule, explain why XOR defeats a linear classifier, describe how hidden
layers and nonlinear activations solve it, trace forward propagation and the idea
of backpropagation, and train and evaluate small neural networks with honest
held-out data.
