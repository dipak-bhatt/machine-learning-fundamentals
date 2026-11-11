# Spam Classification

Classify SMS messages as spam or ham (not spam). This is the text
classification project and the practical application of Naive Bayes.

## Dataset

The SMS Spam Collection is downloaded from GitHub and cached in
`data/raw/sms_spam.tsv`. If the download fails, a small embedded set of labeled
messages is used so the lesson still runs offline.

- About 5,500 unique messages
- Labels: `ham` (0) and `spam` (1)
- Roughly 13% spam, so the classes are imbalanced

## What `train.py` does

1. Cleans the data and removes duplicate messages.
2. Saves exploratory plots:
   - class balance
   - message length by class
3. Splits the data into 80% train and 20% test with stratification.
4. Builds TF-IDF features (unigrams and bigrams) inside a pipeline.
5. Trains and compares:
   - `MultinomialNB` (the main model)
   - logistic regression with balanced class weights
   - linear SVM with balanced class weights
6. Reports accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and 5-fold
   cross-validated F1.
7. Prints and plots the tokens with the strongest spam/ham evidence.
8. Saves confusion matrices, the best model, and metrics.

## Run

```bash
python projects/spam-classification/train.py
```

## Outputs

- `outputs/class_balance_and_lengths.png`
- `outputs/confusion_matrices.png`
- `outputs/top_tokens.png`
- `outputs/metrics.json`
- `models/spam_model.joblib`

## Prior and posterior, in plain words

- The **prior** is the probability of a class before reading the message:
  `P(spam) ~ 0.13` and `P(ham) ~ 0.87` in this dataset. The script prints the
  training priors.
- The **likelihood** is how often each word appears in spam vs. ham. TF-IDF
  weights turn raw counts into scores that emphasize informative words.
- The **posterior** is the updated probability after reading the words:
  `P(spam | words)`. Naive Bayes multiplies the prior by the likelihoods of all
  words (assuming conditional independence) and normalizes the result.

This is why words like "free", "win", and "cash" push a message toward spam,
while conversational words push it toward ham. The top-token plot shows the
learned log-probability differences directly.

## Ideas to extend

- Tune `alpha` for smoothing and `min_df`/`ngram_range` for TF-IDF.
- Add character n-grams to handle obfuscated spam.
- Calibrate probabilities with `CalibratedClassifierCV`.
- Try `ComplementNB`, which often works better on imbalanced text.
