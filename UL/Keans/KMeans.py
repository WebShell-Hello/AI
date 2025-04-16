
# Task 1
# Implement k-means clustering algorithm and cluster the dataset provided using it.
# Vary the value of k from 1 to 9 and compute the Silhouette coefficient for each set of clusters.
# Plot k on the horizontal axis and the Silhouette coefficient on the vertical axis in the same plot.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data source and handle edge cases (like file not provided, file corrupted, or only one data point in the file)
def read_database():
    try:
        data = pd.read_csv('dataset', delim_whitespace=True, header=None)
        if len(data) < 2: # File exists, but there is only 1 sample data point
            print("Error: There is only one data point in the file!!!")
            return
        else: # When data is valid, extract feature columns and convert to numeric type
            features = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').dropna()
            return features  # Return original sample data and feature data
    except FileNotFoundError: # File not found
        print("Error: 'dataset' file not found!!!")
        return
    except pd.errors.EmptyDataError: # File exists, but contains no sample data
        print("Error: The file exists but contains no data!!!")
        return None

# Set random seed to ensure consistent results for each run
np.random.seed(42)  # Reset seed before calling np.random.choice

# Description: Randomly select k rows of sample data as initial centroids without replacement
# x: Sample data set, where x.shape[0] is the total number of samples
# k: Number of clusters
# Logic: Select k samples from x as centroids and return x[indices] as centroids
def initialSelection(x, k):
    row_number = np.random.choice(x.shape[0], size=k, replace=False)
    return x[row_number]

# Calculate the Euclidean distance between two sample data points
def ComputeDistance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Description: Calculate the distance between a point and different centroids to determine which cluster the point belongs to
# x: Sample data set
# centroids: Set of initial centroids
# Logic: Iterate through the sample set, calculating the distance between each data point and each centroid
def assignClusterIds(x, centroids):
    clusters = []
    for point in x: # Iterate through the sample set
        distances = []
        for centroid in centroids: # Iterate through the centroid set
            distance = ComputeDistance(point, centroid)  # Calculate Euclidean distance between current point and centroid
            distances.append(distance)  # Add distance to the list
        cluster_id = np.argmin(distances) # Take the index of the nearest centroid
        clusters.append(cluster_id)
    return np.array(clusters)

# Find the most suitable centroid for each cluster based on the dataset
# Logic: If the cluster is not empty, calculate the mean of all its samples as the new centroid; if empty, assign a new centroid
def computeClusterRepresentatives(x, clusters, k):
    centroids = []
    for i in range(k):
        cluster_points = x[clusters == i]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
        else:
            # Updated logic: For an empty cluster, select the point furthest from other centroids as the new centroid
            distances = np.sum([np.linalg.norm(x - c, axis=1) for c in centroids], axis=0)
            farthest_point_idx = np.argmax(distances)
            centroid = x[farthest_point_idx]
        centroids.append(centroid)
    return np.array(centroids)

# Obtain clusters and centroids for K-means
def kMeans(x, k):
    centroids = initialSelection(x, k) # Obtain initial centroids
    cluster_labels = []
    for i in range(100): # Set an upper limit of 100 iterations to prevent infinite loops
        cluster_labels = assignClusterIds(x, centroids) # Assign samples to clusters based on distances to centroids, reassigning based on updated centroids each iteration
        new_centroids = computeClusterRepresentatives(x, cluster_labels, k) # Update cluster centroids
        if np.allclose(centroids, new_centroids, atol=1e-6): break # Convergence check: Compare old and new centroids; if they are nearly identical, stop iterating
        centroids = new_centroids
    return cluster_labels

# Intra-cluster cohesion: a(i): Average distance from point i to other points in the same cluster. Smaller values indicate tighter clusters.
# Inter-cluster separation: b(i): Minimum average distance from point i to points in other clusters. Larger values indicate better separation.
# Silhouette value: Silhouette = (b-a)/max(a,b) used to evaluate clustering quality
# The silhouette value assesses clustering quality; higher values indicate clearer cluster boundaries and better clustering; negative values suggest failed assignments
def computeSilhouttee(x, clusters):
    silhouette = []
    # Calculate silhouette coefficient for each point
    for i in range(x.shape[0]):
        # Within cluster: Calculate average distance from point i to other points in the same cluster
        inside_cluster = x[clusters == clusters[i]]
        a_i = np.mean([ComputeDistance(x[i], p) for p in inside_cluster if not np.array_equal(p, x[i])]) if len(inside_cluster) > 1 else 0
        # Between clusters: Calculate minimum average distance from point i to points in other clusters
        outside_clusters = [c for c in np.unique(clusters) if c != clusters[i]]
        b_i = min([np.mean([ComputeDistance(x[i], p) for p in x[clusters == c]]) for c in outside_clusters]) if len(outside_clusters) > 0 else 0
        # Calculate silhouette value
        silhouette.append((b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) != 0 else 0)
    # Return the mean silhouette value for k clusters, or 0 if k=1
    return np.mean(silhouette)

# Plot the graph
def plotSilhouttee(k_values, silhouette_scores, x_label, y_label, title, savefig):
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.savefig(savefig)
    plt.show()

# Main function to run the analysis
def main():
    silhouette_scores = []
    k_values = range(1, 10)
    # Obtain and feature values from data source
    features = read_database()
    # Calculate the contour coefficients corresponding to each k value separately
    for k in k_values:
        cluster_labels = kMeans(features.values, k)
        score = computeSilhouttee(features.values, cluster_labels) if k > 1 else 0
        silhouette_scores.append(score)
        print(f"k={k}, Silhouette Score={score:.4f}")
    # Plot results
    x_label = 'Number of Clusters (k)'
    y_label = 'Silhouette Coefficient'
    title = 'Task1: k-Means Clusters Algorithm'
    savefig = 'Task1_k-Means_Silhouette_coefficient.png'
    plotSilhouttee(k_values, silhouette_scores, x_label, y_label, title, savefig)

if __name__ == "__main__":
    main()