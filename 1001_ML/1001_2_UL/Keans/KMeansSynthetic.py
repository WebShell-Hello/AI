
# Task 2
# Generate synthetic data of same size (i.e. same number of data points) as the dataset provided and use this data to cluster K Means.
# Plot k in the horizontal axis and the Silhouette coefficient in the vertical axis in the same plot.

import numpy as np
from KMeans import (read_database, kMeans, computeSilhouttee, plotSilhouttee)

# Fix random seed for reproducibility
np.random.seed(42)

def generateSyntheticData(original_data):
    # Calculate mean for each column of the original data
    means = np.mean(original_data, axis=0)
    # Calculate standard deviation for each column of the original data
    stds = np.std(original_data, axis=0)
    # Check if standard deviation is zero to avoid generating invalid data
    # (since normal distribution requires standard deviation > 0)
    stds[stds == 0] = 1e-10  # Set features with zero std to a very small value
    # Sample using original data's mean as distribution center
    # and processed std as distribution scale
    return np.random.normal(loc=means, scale=stds, size=original_data.shape)

def main():
    silhouette_scores = []
    k_values = range(1, 10)  # k from 1 to 9
    features = read_database()  # Read dataset and extract features
    synthetic_data = generateSyntheticData(features)  # Generate synthetic data
    # Calculate the contour coefficients corresponding to each k value separately
    for k in k_values:
        cluster_labels = kMeans(synthetic_data, k)
        score = computeSilhouttee(synthetic_data, cluster_labels) if k > 1 else 0
        silhouette_scores.append(score)
        print(f"k={k}, Silhouette Score={score:.4f}")
    # Plot results
    x_label = 'Number of Clusters (k)'
    y_label = 'Silhouette Coefficient'
    title = 'Task2: k-Means Clusters Algorithm with Synthetic Data'
    savefig = 'Task2_k-Means_Silhouette_coefficient_SyntheticData.png'
    plotSilhouttee(k_values, silhouette_scores, x_label, y_label, title, savefig)

if __name__ == "__main__":
    main()