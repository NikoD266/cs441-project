import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)



data = pd.read_csv("C:\\Users\\ejpot\\Downloads\\little_data.csv")
trimmed_data = data.iloc[:, 1:]
print(trimmed_data.head())

scaled_df = StandardScaler().fit_transform(trimmed_data)

# initialize kmeans parameters
kmeans_kwargs = {
"init": "random",
"n_init": 10,
"random_state": 1,
}

# create list to hold SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_df)
    sse.append(kmeans.inertia_)

# visualize results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

from sklearn.decomposition import PCA

# Reduce to 2D using PCA
pca = PCA(n_components=2)
pca_df = pca.fit_transform(scaled_df)

# Apply KMeans on PCA-reduced data
kmeans = KMeans(n_clusters=4, init="random", n_init=10, random_state=1)
kmeans.fit(pca_df)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualize clusters
plt.figure(figsize=(8, 6))
plt.scatter(pca_df[:, 0], pca_df[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("KMeans Clustering on PCA-Reduced Data (k=4)")
plt.legend()
plt.grid(True)
plt.show()

trimmed_data['Cluster'] = labels  # from your KMeans model

cluster_summary = trimmed_data.groupby('Cluster').agg(['mean', 'std'])
print(cluster_summary)

import matplotlib.pyplot as plt

attributes = trimmed_data.columns
num_attributes = len(attributes)

# Set up subplot grid
rows = (num_attributes + 2) // 3
fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
axes = axes.flatten()

# Plot each attribute
for i, attr in enumerate(attributes):
    for cluster in sorted(trimmed_data['Cluster'].unique()):
        subset = trimmed_data[trimmed_data['Cluster'] == cluster]
        axes[i].scatter(subset.index, subset[attr], label=f"Cluster {cluster}", alpha=0.6, s=6)
    axes[i].set_title(attr)
    axes[i].grid(True)

# Hide unused subplots if any
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout(pad=7.0)

plt.show()

# Count the number of samples in each cluster
cluster_counts = trimmed_data['Cluster'].value_counts().sort_index()

# Plot counted samples as a bar chart
plt.figure(figsize=(6, 4))
plt.bar(cluster_counts.index, cluster_counts.values, color='skyblue', edgecolor='black')
plt.xlabel("Cluster")
plt.ylabel("Number of Samples")
plt.title("Counts per Cluster")
plt.xticks(cluster_counts.index)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
