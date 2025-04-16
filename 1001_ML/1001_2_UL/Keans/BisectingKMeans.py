
# Task 4
# Implement the Bisecting k-Means algorithm to compute a hierarchy of clusterings that refines the initial single cluster to 9 clusters.
# For each s from 1 to 9, extract from the hierarchy of clusterings the clustering with s clusters and compute the Silhouette coefficient for this clustering.
# Plot s in the horizontal axis and the Silhouette coefficient in the vertical axis in the same plot.

import numpy as np
from KMeans import (read_database, kMeans, computeSilhouttee, plotSilhouttee)

def bisecting_kmeans(data, max_clusters):
    # Initialize with all points in cluster of k=1
    clusterings = [np.zeros(data.shape[0], dtype=int)]  # k=1
    for k in range(2, max_clusters + 1):
        # Use a tournament-like algorithm to find the best split
        # best_clustering stores the current "best clustering scheme"
        # best_score is initialized to negative infinity to represent the initial best score
        best_clustering = None
        best_score = -np.inf

        # For each existing cluster, try to split it
        for cluster_id in np.unique(clusterings[-1]):
            # Get all points in current cluster
            points = (clusterings[-1] == cluster_id)
            cluster_data = data[points]

            if len(cluster_data) < 2: continue  # Skip clusters with only 1 point
            sub_clusters = kMeans(cluster_data, 2)  # Split cluster using k-Means
            # Update labels for new clusters (e.g., split 0 into 0 and 1)
            new_clustering = clusterings[-1].copy()
            new_clustering[points] = (cluster_id * 2) + sub_clusters
            score = computeSilhouttee(data, new_clustering)  # Calculate silhouette score
            # Keep the best split
            if score > best_score:
                best_score = score
                best_clustering = new_clustering
        clusterings.append(best_clustering if best_clustering is not None else clusterings[-1].copy())

    return clusterings

def main():
    silhouette_scores = []  # Silhouette score for k=1 is 0 by definition
    k_values = range(1, 10)
    # Read data
    features = read_database()
    # Execute Bisecting K-Means
    clusterings = bisecting_kmeans(features.values, max(k_values))
    # Compute Silhouette coefficients
    for k in k_values:
        # Cluster index starts from 0, so index = k-1
        cluster_labels = clusterings[k - 1]
        score = computeSilhouttee(features.values, cluster_labels) if k > 1 else 0
        silhouette_scores.append(score)
        print(f"s={k}, Silhouette Score={score:.4f}")
    # Plot results
    x_label = 'Number of Clusters (s)'
    y_label = 'Silhouette Coefficient'
    title = 'Task4: Bisecting k-Means Clusters Algorithm'
    savefig = 'Task4_Bisecting_k-Means_Silhouette_coefficient.png'
    plotSilhouttee(k_values, silhouette_scores, x_label, y_label, title, savefig)

if __name__ == "__main__":
    main()