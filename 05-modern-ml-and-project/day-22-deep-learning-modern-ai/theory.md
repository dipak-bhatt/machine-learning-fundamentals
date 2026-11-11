# Day 22 — Theory: Deep Learning and Modern AI

**Module:** Module 6: Modern ML and Project

Deep learning is the part of machine learning built on multilayer neural networks. It
powers image recognition, speech transcription, translation, and the large language
models behind modern chatbots. It is also surrounded by more hype than any other area
of the field.

This day explains what neural networks actually compute, how they are trained, and
where they fit relative to the classical models from earlier weeks. The goal is not to
turn you into a deep-learning engineer in one day. It is to give you a correct mental
model, so you can judge when a neural network is the right tool and when it is
overkill.

## In Plain English

**Think of it like this:** deep learning builds its own features by stacking
simple neuron layers, instead of relying on features a human designed.

- Neurons → layers → networks; **nonlinear activations** let the stack learn
  complex patterns.
- **CNNs** suit images; **RNNs and transformers** suit sequences and text.
- Training uses a **loss + backpropagation + gradient descent**, often on GPUs.
- It needs lots of data and compute; on small tables, classical ML often wins.
- Watch for overfitting, bias, and the cost of very large models.

## 1. Traditional ML versus Deep Learning

Classical models rely on **hand-engineered features**. A logistic regression on house
prices uses area, age, and location directly. A random forest on tabular data splits
on columns a human chose. The model learns a mapping from those features to the
target, but the representation is given.

Deep learning learns the representation too. A convolutional network for images starts
from pixels and learns filters for edges, textures, and shapes. A transformer for text
learns embeddings and attention patterns. The features are not designed by hand; they
are learned from data.

| Aspect | Classical ML | Deep learning |
|--------|--------------|---------------|
| Features | engineered | learned |
| Data needs | modest | large |
| Compute | CPU often enough | GPU/TPU common |
| Interpretability | usually higher | usually lower |
| Tabular data | strong | often not better |
| Images, audio, text | limited | state of the art |
| Training time | seconds to minutes | minutes to weeks |

The most important practical fact: **on small or medium tabular datasets, classical
models are often as good as or better than neural networks**, and far cheaper. Deep
learning earns its cost when data are large, raw, and high-dimensional, especially
images, audio, and text.

## 2. The Neuron

A single artificial neuron computes a weighted sum of its inputs plus a bias, then
applies a nonlinear function:

```text
z = w1*x1 + w2*x2 + ... + wn*xn + b
a = activation(z)
```

- `x` are the inputs.
- `w` are the weights, learned from data.
- `b` is the bias, also learned.
- `activation` is a nonlinear function.

Without the nonlinearity, stacking layers would collapse into a single linear
transformation. The activation is what lets a network represent curves and
interactions.

## 3. From Neuron to Network

A **multilayer perceptron (MLP)** organizes neurons into layers:

- **Input layer**: one node per feature.
- **Hidden layers**: one or more layers of neurons.
- **Output layer**: one node for regression or binary classification, or one per class
  for multi-class classification.

```text
input -> hidden -> hidden -> output
  x1  ->  h1  ->   h1  ->   y
  x2  ->  h2  ->   h2
  x3  ->  h3  ->   h3
```

Each layer's outputs become the next layer's inputs. A network with at least one
hidden layer is a **deep** network in the modern sense, though "deep" usually means
several layers.

The **capacity** of the network — how complex a function it can represent — grows with
the number of layers and neurons. More capacity can fit more complex patterns and also
overfit more easily.

## 4. Activation Functions

### ReLU

```text
relu(z) = max(0, z)
```

The default for hidden layers. Cheap, and it avoids the vanishing-gradient problem
that plagued older activations. Its derivative is 1 for positive inputs and 0
otherwise, so gradients flow well.

### Sigmoid

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

Squashes values to (0, 1). Historically common, now mostly used for binary output
probabilities. Saturates for large positive or negative inputs, which can slow
learning.

### Tanh

```text
tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
```

Squashes to (-1, 1), zero-centred. Sometimes works better than sigmoid in hidden
layers but still saturates.

### Softmax

For multi-class output, converts a vector of scores into probabilities that sum to 1:

```text
softmax(z_i) = exp(z_i) / sum_j exp(z_j)
```

Used with cross-entropy loss for classification.

### Choosing

- Hidden layers: ReLU is the default; tanh is a reasonable alternative.
- Binary output: sigmoid.
- Multi-class output: softmax.
- Regression output: no activation (linear).

## 5. Loss Functions

The loss measures how wrong a prediction is. Training minimizes it.

### Mean squared error (regression)

```text
MSE = (1/n) * sum (y_i - y_hat_i)^2
```

Penalizes large errors heavily. Sensitive to outliers.

### Mean absolute error (regression)

```text
MAE = (1/n) * sum |y_i - y_hat_i|
```

More robust to outliers, but its gradient is constant and less smooth.

### Binary cross-entropy (binary classification)

```text
BCE = -(1/n) * sum [ y_i*log(p_i) + (1-y_i)*log(1-p_i) ]
```

Works with a sigmoid output and predicted probabilities `p`.

### Categorical cross-entropy (multi-class)

```text
CCE = -(1/n) * sum_i sum_c y_ic * log(p_ic)
```

Works with a softmax output. The loss is large when the model assigns low probability
to the correct class.

Choosing the loss is choosing what "wrong" means. Cross-entropy for classification
and squared error for regression are the defaults.

## 6. The Forward Pass

Training begins with a forward pass: inputs flow through the layers to produce an
output. For a two-layer network:

```text
h1 = relu(X @ W1 + b1)
h2 = relu(h1 @ W2 + b2)
y_hat = softmax(h2 @ W3 + b3)
loss = cross_entropy(y, y_hat)
```

The weights `W` and biases `b` start random. The loss tells us how far the prediction
is from the target.

## 7. Backpropagation

**Backpropagation** computes how much each weight contributed to the loss, using the
chain rule of calculus.

The idea:

1. Compute the loss.
2. Compute the gradient of the loss with respect to the output.
3. Propagate the gradient backward through each layer.
4. Obtain the gradient for every weight and bias.

For a weight `w`, the gradient `dLoss/dw` tells us the slope: increasing `w` slightly
changes the loss by about that amount. We then move `w` in the opposite direction:

```text
w = w - learning_rate * dLoss/dw
```

This is **gradient descent**. Backpropagation is just the efficient algorithm for
computing all those gradients in one backward pass.

You do not need to derive the chain rule by hand to use deep learning, but you should
know that training is gradient-based optimization, not a search over discrete rules.

## 8. Learning Rate and Optimizers

The **learning rate** controls the step size.

- Too small: training is slow and may stall.
- Too large: the loss oscillates or diverges.
- Just right: steady decrease.

Optimizers improve on plain gradient descent:

- **SGD**: stochastic gradient descent, updates on mini-batches.
- **Momentum**: accumulates past gradients to smooth the path.
- **RMSProp**: adapts the step size per parameter.
- **Adam**: combines momentum and adaptive step sizes; a strong default.

Optimizers do not change the objective, only how we move toward its minimum.

## 9. Epochs, Batches, and Iterations

- **Batch size**: number of examples used to compute one gradient update.
- **Iteration**: one update.
- **Epoch**: one full pass over the training set.

With 1,000 examples and batch size 100, one epoch is 10 iterations. Small batches give
noisier but more frequent updates; large batches give smoother but slower updates.

Noise from mini-batches can help generalization, acting like a mild regularizer.

## 10. Overfitting and Regularization

Neural networks have many parameters and can memorize training data. Defenses:

- **More data**: the most effective fix.
- **Dropout**: randomly zero a fraction of activations during training, forcing the
  network not to rely on any single path.
- **Weight decay (L2)**: penalize large weights.
- **Early stopping**: stop when validation loss stops improving.
- **Data augmentation**: create transformed copies of training examples.
- **Smaller networks**: reduce capacity.
- **Batch normalization**: stabilize and speed training, with a mild regularizing
  effect.

Always monitor a validation set. Training loss falling while validation loss rises is
the signature of overfitting.

## 11. Convolutional Neural Networks

For images, a fully connected layer on raw pixels is wasteful. A **convolutional
neural network (CNN)** uses small filters that slide across the image:

- **Convolution**: detects local patterns such as edges and textures.
- **Pooling**: downsamples, reducing spatial size.
- **Weight sharing**: the same filter is used at every position, drastically reducing
  parameters.
- **Hierarchy**: early layers detect edges, deeper layers detect shapes and objects.

CNNs encode the assumption that nearby pixels matter more than distant ones and that
the same pattern can appear anywhere. That inductive bias is why they need less data
than an MLP for images.

## 12. Sequence Models

For text, audio, and time series, order matters. Early sequence models used recurrent
neural networks:

- **RNN**: processes one element at a time, carrying a hidden state.
- **LSTM**: adds gates to remember and forget over long spans.
- **GRU**: a simpler gated variant.

Recurrent models struggle with very long sequences and are slow to train because they
are inherently sequential. Transformers replaced them for most language tasks.

## 13. Transformers and Large Language Models

A **transformer** processes a sequence as a set of tokens and uses **self-attention**
to let each token weigh every other token. Attention captures long-range relationships
without recurrence and parallelizes well on modern hardware.

A **large language model (LLM)** is a transformer trained on enormous text corpora to
predict the next token. Training happens in stages:

- **Pretraining**: learn general language patterns from vast unlabeled text.
- **Fine-tuning**: adapt to a specific task with labeled examples.
- **Instruction tuning and alignment**: teach the model to follow instructions and
  behave safely.

At inference, an LLM generates text one token at a time, conditioned on the prompt and
its parameters. **Prompting** steers behavior without changing weights. **Retrieval-
augmented generation (RAG)** supplies external documents so the model can answer with
current or private information.

LLMs are powerful but not databases. They can produce fluent falsehoods, reflect
biases in their training data, and are expensive to train and run. Use them with
verification, clear scope, and human oversight.

## 14. Generative versus Predictive ML

- **Predictive ML** estimates a target from inputs: will this customer churn, what is
  this house worth, is this email spam.
- **Generative AI** learns a distribution and produces new content: text, images,
  audio, code.

The two overlap. A language model is predictive at the token level (next-token
prediction) and generative at the system level. A classifier is purely predictive.

Generative systems raise different risks: fabricated content, copyright, impersonation,
and the difficulty of evaluating open-ended output. Evaluation is not a single
accuracy number; it requires human judgment, task-specific checks, and red-teaming.

## 15. Compute, Data, and Limitations

Deep learning is expensive:

- **Data**: large labeled datasets are costly to collect and clean.
- **Compute**: GPUs and TPUs consume energy and money.
- **Talent and tooling**: debugging is harder than for linear models.
- **Latency**: large models may be too slow for real-time use.
- **Carbon**: training and serving have an environmental cost.

Limitations:

- **No guaranteed correctness.** Neural networks interpolate; they do not reason
  symbolically.
- **Distribution shift.** Performance drops when deployment data differ from training.
- **Opacity.** Explanations are approximate.
- **Brittleness.** Small input changes can cause large output changes.
- **Evaluation difficulty.** Open-ended generative output is hard to score.
- **Reproducibility.** Nondeterminism and hardware differences complicate exact
  reproduction.

None of this makes deep learning unusable. It means the same discipline applies:
baselines, held-out evaluation, leakage checks, monitoring, and honest communication.

## 16. Common Mistakes

- **Using a neural network on a small tabular dataset without comparing to a simple
  model.** Often no gain, much more cost.
- **Forgetting to scale inputs.** Neural networks are sensitive to feature scale.
- **Using the wrong output activation or loss.** Sigmoid with MSE for classification,
  for example.
- **No validation set.** Overfitting goes unnoticed.
- **Learning rate too high or too low.** Loss diverges or crawls.
- **Early stopping on a tiny validation split.** It may stop before the model learns.
- **Judging by training loss.** It always improves; validation loss is what matters.
- **Ignoring class imbalance.** The network predicts the majority class.
- **Treating LLM output as fact.** Verify, ground, and constrain.
- **Skipping cost and latency analysis.** A good model that is too slow is not
  deployable.

## 17. Summary

- Deep learning learns representations; classical ML uses engineered features.
- Neural networks stack nonlinear layers of weighted sums.
- ReLU, sigmoid, tanh, and softmax are the common activations.
- Loss functions define what "wrong" means; cross-entropy for classification, squared
  error for regression.
- Backpropagation computes gradients; optimizers use them to update weights.
- Learning rate, batch size, and epochs control training dynamics.
- Regularization fights overfitting: dropout, weight decay, early stopping,
  augmentation.
- CNNs exploit spatial structure; transformers exploit attention over sequences.
- LLMs are transformers pretrained on massive text and adapted to tasks.
- Generative AI produces content; predictive ML estimates targets.
- Deep learning costs data, compute, and energy, and it is not always better.
- Rigorous evaluation and responsible use remain essential.

## Outcome

Learners can explain how a neural network computes and learns, choose activations and
losses, describe backpropagation and gradient descent, tune learning rate and batch
size, apply regularization, distinguish CNNs, sequence models, and transformers,
explain what LLMs are and are not, and decide when deep learning is worth its cost.
