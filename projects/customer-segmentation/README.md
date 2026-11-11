# Customer Segmentation

Group customers into segments with K-means clustering. This is the
unsupervised-learning project and focuses on choosing the number of clusters.

## Dataset

The Mall Customers dataset is downloaded from GitHub and cached in
`data/raw/mall_customers.csv`. If the download fails, the script generates a
deterministic synthetic customer dataset with five underlying segments.

Features used for clustering:

- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

## What `train.py` does

1. Cleans the data and prints summary statistics.
2. Saves exploratory plots:
   - feature distributions
   - income vs. spending score before clustering
3. Scales the three features with `StandardScaler`.
4. Runs K-means for `k = 2..10` and records:
   - inertia (for the elbow method)
   - silhouette score
5. Picks the `k` with the highest silhouette score and fits the final model.
6. Saves the cluster plot with centroids, a labeled customer CSV, and cluster
   profile tables.
7. Saves the fitted model and scaler with joblib.

## Run

```bash
python projects/customer-segmentation/train.py
```

## Outputs

- `outputs/feature_distributions.png`
- `outputs/customers_raw.png`
- `outputs/cluster_selection.png` (elbow + silhouette)
- `outputs/customer_segments.png`
- `outputs/customer_segments.csv`
- `outputs/cluster_profiles.csv`
- `outputs/cluster_selection.json`
- `models/segmentation_model.joblib`

## Reading the results

- **Elbow method:** look for the point where inertia stops dropping quickly.
- **Silhouette score:** values closer to 1 mean clusters are dense and well
  separated; values near 0 mean overlapping clusters.

The two methods do not always agree. The script chooses the silhouette
maximum, but the elbow plot is printed alongside it so you can judge both.

## Segments as a downstream feature

Cluster labels are not the end goal. They can become a feature for supervised
models (for example, churn or campaign response), be used for targeted
marketing, or be combined with cluster profiles to name segments such as
"high income, low spenders". Always validate segments on new data before
acting on them.

## Ideas to extend

- Compare K-means with hierarchical clustering and DBSCAN (Day 16).
- Reduce dimensions with PCA and plot the clusters in 2D (Day 17).
- Choose `k` with `silhouette_score` plus a business constraint on segment size.
