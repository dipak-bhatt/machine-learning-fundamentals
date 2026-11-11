# Customer Churn Prediction

Predict whether a telecom customer will churn. This is the end-to-end binary
classification project, with emphasis on class imbalance and metric choice.

## Dataset

The IBM Telco Customer Churn dataset is downloaded from GitHub and cached in
`data/raw/telco_churn.csv`. If the download fails, the script generates a
deterministic synthetic telco-style dataset with the same column types, so the
lesson still runs offline.

- About 7,000 customers
- Numeric and categorical features (contract, internet service, payment method,
  tenure, monthly charges, ...)
- Target: `Churn` (Yes/No), roughly 27% positive

## What `train.py` does

1. Cleans the data:
   - drops the customer ID
   - converts `TotalCharges` to numeric
   - maps `Churn` to 0/1
2. Explores the data and saves plots:
   - class balance
   - churn rate by contract type
   - tenure and monthly charges split by churn
3. Builds a `ColumnTransformer` pipeline:
   - median imputation + scaling for numeric columns
   - most-frequent imputation + one-hot encoding for categorical columns
4. Trains and compares:
   - `DummyClassifier(strategy="prior")` baseline
   - logistic regression with `class_weight="balanced"`
   - random forest with balanced subsampling
   - gradient boosting
5. Handles imbalance through class weights and by evaluating precision,
   recall, F1, ROC-AUC, and PR-AUC instead of accuracy alone.
6. Saves confusion matrices, the classification report, and the best model.

The model is selected by F1 on the churn class, which is a more useful score
than accuracy when churners are the minority.

## Run

```bash
python projects/customer-churn-prediction/train.py
```

## Outputs

- `outputs/class_balance.png`
- `outputs/churn_by_contract.png`
- `outputs/tenure_and_charges.png`
- `outputs/confusion_matrices.png`
- `outputs/metrics.json`
- `models/churn_model.joblib`

## Why accuracy is misleading

If 73% of customers stay, a model that always predicts "no churn" already
scores 0.73 accuracy but has zero recall for churners. Precision, recall, F1,
and PR-AUC expose that failure, which is exactly what the baseline is there to
demonstrate.

## Ideas to extend

- Tune the decision threshold to trade precision against recall.
- Add `imblearn` SMOTE inside the pipeline and compare.
- Use permutation importance to explain the best model.
- Serve the model with the Day 23 FastAPI deployment lesson.
