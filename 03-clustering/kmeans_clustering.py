import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Apply Seaborn style formatting
sns.set_theme()

# 1. Generate synthetic blobs dataset
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)

# 2. Fit K-Means clustering model (fixed random_state for reproducibility)
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)

# 3. Predict clusters and extract centroids
predictions = kmeans.predict(X)
centers = kmeans.cluster_centers_

# 4. Plot clustered data points with centroids overlay
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=predictions, s=30, cmap="cool")
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c="black",
    s=200,
    alpha=0.75,
    marker="X",
    label="Centroids",
)
plt.title("K-Means Clustering (k=4)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.show()
