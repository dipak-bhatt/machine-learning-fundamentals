# Day 22 — Deep Learning and Modern AI Overview

**Module:** Module 6: Modern ML and Project

Deep learning is the technology behind modern image recognition, speech systems,
translation, and large language models. It is powerful, expensive, and often applied
where a simpler model would do better. This day explains what neural networks compute,
how they learn, and where they fit in the broader landscape of machine learning.

The practical builds a small multilayer perceptron on an 8x8 image dataset and compares
it honestly with logistic regression and a random forest. The result is instructive:
the network can match the classical models, but on this small dataset it does not
meaningfully beat them.

## What You Will Learn

- How deep learning differs from classical machine learning.
- What a neuron and a multilayer perceptron compute.
- The roles of activation functions and loss functions.
- How backpropagation and gradient descent train a network.
- How learning rate, batch size, and epochs affect training.
- How dropout, weight decay, and early stopping reduce overfitting.
- What CNNs and sequence models are designed to handle.
- What transformers and large language models are, at a high level.
- The difference between generative AI and predictive ML.
- The data, compute, and environmental costs of deep learning.
- When to use a neural network and when to stay classical.

## Detailed Content

- [Theory](theory.md) — traditional ML versus deep learning, neurons, MLPs,
  activations, losses, the forward pass, backpropagation, learning rate, optimizers,
  epochs and batches, regularization, CNNs, sequence models, transformers, LLMs,
  generative versus predictive AI, costs, limitations, and common mistakes.
- [Practical](practical.md) — load the digits dataset, compare logistic regression and
  random forest with cross-validation, train a small MLP with and without early
  stopping, plot the loss curve, evaluate once on test, inspect the confusion matrix
  and a misclassified image, and review a PyTorch and transformer sketch.

## Files

No CSV is needed. The practical uses:

- **Digits** (`load_digits`) — 1,797 images of 8x8 pixels, 10 classes, built into
  scikit-learn.
- scikit-learn's `MLPClassifier` for the neural network.

Optional reference snippets use `torch` and `transformers`, which may not be installed.
They are for illustration and are not required to complete the practical.

## Before You Start

- Day 21: end-to-end pipelines and honest evaluation.
- Day 20: pipelines and hyperparameter tuning.
- Day 18 and Day 19: classification metrics, cross-validation, and leakage.
- Day 11: ensemble models as strong tabular baselines.

## Outcome

Learners can explain how a neural network computes and learns, describe activations,
losses, backpropagation, and optimizers, train a small MLP, compare it fairly with
classical models, read a training curve and confusion matrix, and judge when deep
learning is worth its data, compute, and maintenance costs.
