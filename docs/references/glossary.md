# Plain-English Glossary

Short definitions of the terms used across the course. Each entry explains the
idea in everyday words first; the formal detail is in the lesson where the term
appears.

## Data and Foundations

- **Dataset** — a table of examples. Rows are cases, columns are pieces of
  information.
- **Row / sample / example / observation** — one case in the table, such as one
  customer or one house.
- **Column / feature / variable / attribute** — one piece of information, such
  as age or price.
- **Target / label / output** — the column the model is trying to predict.
- **Features (X)** — the input columns used to make a prediction.
- **Numerical data** — numbers (age, price) that can be continuous or discrete.
- **Categorical data** — groups or labels (city, color) that are nominal or
  ordinal.
- **Mean** — the average; sensitive to extreme values.
- **Median** — the middle value after sorting; safer when there are outliers.
- **Mode** — the most frequent value; useful for categories.
- **Variance** — the average squared distance from the mean; measures spread.
- **Standard deviation** — the square root of variance; spread in the original
  units.
- **Percentile** — the value below which a given percentage of the data falls.
- **Outlier** — a value far away from the rest; check whether it is a mistake or
  a genuine rare case.
- **Distribution** — the shape of how often values appear (bell-shaped, skewed,
  uniform, and so on).
- **Population** — every case we care about.
- **Sample** — the smaller set we actually measured.
- **Sampling bias** — when the sample does not fairly represent the population.
- **Correlation** — two variables tending to move together; **not** proof that
  one causes the other.
- **Confounding variable** — a hidden third factor that drives two correlated
  variables.
- **Probability** — a number between 0 and 1 saying how likely an event is.
- **Conditional probability** — the chance of one event given that another has
  happened.
- **Noise** — random error in the data that no model can fully remove.

## Training and Evaluation

- **Supervised learning** — learning from examples that include the correct
  answer.
- **Unsupervised learning** — finding structure in data that has no labels.
- **Reinforcement learning** — learning by trial, error, and reward.
- **Training set** — the data the model learns from.
- **Validation set** — the data used to compare and tune models.
- **Test set** — the untouched data used for one final honest check.
- **Generalization** — doing well on new, unseen data, not just the training
  data.
- **Overfitting** — memorizing the training data, including its noise, and
  failing on new data.
- **Underfitting** — being too simple to learn the real pattern.
- **Bias** — error from a model that is too simple or makes the wrong
  assumption.
- **Variance** — error from a model that is too sensitive to the particular
  training sample.
- **Bias-variance tradeoff** — reducing one kind of error often increases the
  other; aim for the balance.
- **Data leakage** — information from the test set or the future sneaking into
  training.
- **Cross-validation** — testing on several different splits so every row gets a
  turn as validation data.
- **Accuracy** — the fraction of predictions that are correct; misleading on
  imbalanced data.
- **Precision** — of the cases the model flagged as positive, the fraction that
  really are positive.
- **Recall** — of the real positive cases, the fraction the model found.
- **F1 score** — a single score balancing precision and recall.
- **Confusion matrix** — a table of true/false positives and negatives.
- **ROC curve** — a plot of true-positive rate against false-positive rate across
  thresholds.
- **AUC** — the area under the ROC curve; 0.5 is random guessing, 1.0 is perfect.
- **Threshold** — the probability cutoff used to turn a score into a class label.

## Models

- **Regression** — predicting a number.
- **Classification** — predicting a category.
- **Linear regression** — fitting a straight line to predict a number.
- **Least squares** — choosing the line with the smallest squared errors.
- **R²** — the fraction of the target's variation that the model explains.
- **RMSE / MAE** — typical prediction error, in the target's units.
- **Gradient descent** — improving parameters in small steps by walking downhill
  on the error.
- **Learning rate** — the step size in gradient descent.
- **Regularization** — penalizing large weights to prevent overfitting.
- **Ridge / Lasso** — regularized linear models; Lasso can zero out features.
- **Logistic regression** — a linear model whose output is squashed into a
  probability for classification.
- **Sigmoid** — the S-shaped function that turns any score into a probability
  between 0 and 1.
- **Log loss / cross-entropy** — the classification loss that punishes confident
  wrong answers.
- **Perceptron** — the simplest trainable neuron; draws a straight boundary.
- **Activation function** — the nonlinear step applied inside a neuron (step,
  sigmoid, tanh, ReLU).
- **Neural network** — layers of neurons that learn increasingly complex
  patterns.
- **Hidden layer** — a layer between the input and output that builds new
  features.
- **Forward propagation** — computing a prediction from inputs to output.
- **Backpropagation** — sending the error backward to update every weight.
- **Epoch** — one full pass through the training data.
- **Decision tree** — a flowchart of yes/no questions ending in a prediction.
- **Impurity (Gini / entropy)** — a measure of how mixed the classes are in a
  tree node.
- **Pruning** — limiting a tree's growth so it does not memorize noise.
- **Ensemble** — combining several models, usually by voting or averaging.
- **Bagging / random forest** — many models trained on resampled data, then
  combined.
- **Boosting / gradient boosting** — models trained in sequence, each correcting
  the previous one's errors.
- **Support vector machine (SVM)** — a classifier that maximizes the gap
  (margin) between classes.
- **Margin** — the empty band around an SVM's boundary.
- **Support vectors** — the training points closest to the boundary; they define
  it.
- **Kernel** — a similarity function that lets an SVM draw curved boundaries.
- **Naive Bayes** — a fast classifier that assumes features are independent
  given the class.
- **Prior / posterior** — belief before evidence / updated belief after
  evidence.
- **Bayesian network** — a graph of variables and their direct influences, with
  probability tables.
- **K-nearest neighbors (KNN)** — predicting from the `k` most similar training
  examples.
- **Lazy learning** — doing the work at prediction time instead of training time.

## Unsupervised Learning

- **Clustering** — grouping similar cases without labels.
- **K-means** — clustering by repeatedly assigning points to the nearest center
  and moving the centers.
- **Centroid** — the center of a cluster.
- **Elbow method** — choosing `k` where adding another cluster stops helping
  much.
- **Silhouette score** — a measure of how well-separated clusters are.
- **Hierarchical clustering** — merging or splitting groups to build a tree of
  clusters.
- **Dendrogram** — the tree diagram showing which groups merged at which
  distance.
- **DBSCAN** — density-based clustering that finds dense clumps and labels sparse
  points as noise.
- **Dimensionality reduction** — replacing many features with a smaller set that
  keeps most of the information.
- **PCA** — finding the directions of greatest variance and projecting data onto
  them.
- **Principal component** — one of PCA's new axes.
- **Explained variance** — the fraction of total variation captured by a
  component.
- **Scree plot** — a chart of explained variance per component, used to choose
  how many to keep.

## Projects and Deployment

- **Pipeline** — preprocessing and model bundled into one object so the same
  steps always run.
- **ColumnTransformer** — applying different preprocessing to different columns.
- **Imputation** — filling in missing values.
- **One-hot encoding** — turning a categorical column into several 0/1 columns.
- **Scaling / standardization** — putting numeric features on comparable ranges.
- **Hyperparameter** — a setting chosen before training (for example, `k` or
  `C`), not learned from data.
- **Grid search** — trying a defined set of hyperparameter combinations.
- **Random search** — sampling hyperparameter combinations at random.
- **API** — a service endpoint other software can call, often over HTTP.
- **Model deployment** — making a trained model available to real users or
  systems.
- **Data drift / concept drift** — the real world changing so an old model
  becomes less accurate.
- **Monitoring** — watching a deployed model for errors, slowdowns, and drift.
