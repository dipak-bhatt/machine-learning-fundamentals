# Projects

End-to-end machine learning projects that apply the course material. Each
project is self-contained: a `train.py` script, a `README.md` with the workflow
and expected outputs, and generated artifacts under `outputs/` and `models/`.

| Project | Task | Main algorithms | Dataset |
|---------|------|-----------------|---------|
| [house-price-prediction](house-price-prediction/) | Regression | Linear Regression, Ridge | California Housing (scikit-learn) |
| [customer-churn-prediction](customer-churn-prediction/) | Binary classification | Dummy baseline, Logistic Regression, Random Forest, Gradient Boosting | IBM Telco Churn |
| [customer-segmentation](customer-segmentation/) | Clustering | K-means, elbow method, silhouette | Mall Customers |
| [spam-classification](spam-classification/) | Text classification | Multinomial Naive Bayes, Logistic Regression, Linear SVM | SMS Spam Collection |

## Running a project

From the repository root:

```bash
python projects/house-price-prediction/train.py
python projects/customer-churn-prediction/train.py
python projects/customer-segmentation/train.py
python projects/spam-classification/train.py
```

Install dependencies first:

```bash
pip install -r requirements.txt
```

## Notes

- External datasets are downloaded once and cached in each project's
  `data/raw/` directory. Churn, segmentation, and spam scripts fall back to
  deterministic synthetic or embedded data if the download fails.
- Generated plots, metrics, and models are written to `outputs/` and `models/`.
  These directories are git-ignored.
- Every script uses `RANDOM_STATE = 42` so results are reproducible.
