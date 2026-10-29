import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

X, _ = make_blobs(n_samples=500, centers=4, random_state=42)
X = StandardScaler().fit_transform(X)
best_k = 4
model = KMeans(n_clusters=best_k, n_init=10, random_state=42)
labels = model.fit_predict(X)
print("Inertia:", model.inertia_)
print("Silhouette:", silhouette_score(X, labels))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=20)
plt.savefig("clusters.png")
