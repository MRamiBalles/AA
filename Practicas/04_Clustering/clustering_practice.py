import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

def plot_kmeans(X, n_clusters, filename):
    """
    Runs K-Means and plots the results with centroids.
    """
    print(f"Running K-Means with K={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X)
    
    score = silhouette_score(X, y_kmeans)
    print(f"Silhouette Score (K={n_clusters}): {score:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis', label='Data Points')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')
    plt.title(f"K-Means Clustering (K={n_clusters}, Silhouette={score:.2f})")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    print(f"Saved plot: {filename}")

def plot_dendrogram_chart(X, filename):
    """
    Computes and plots the Hierarchical Clustering Dendrogram.
    """
    print("Computing Hierarchical Clustering (Linkage)...")
    # 'ward' minimizes variance within clusters
    Z = linkage(X, method='ward')
    
    plt.figure(figsize=(10, 7))
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance (Ward)")
    
    dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
    plt.grid(True)
    plt.savefig(filename)
    print(f"Saved plot: {filename}")

def main():
    # Generate Synthetic Data (Blobs)
    print("Generating synthetic data...")
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    
    # --- Part 1: K-Means ---
    # Try with K=4 (Correct number)
    plot_kmeans(X, 4, "clustering_kmeans_k4.png")
    
    # Try with K=2 (Incorrect number, to see difference)
    plot_kmeans(X, 2, "clustering_kmeans_k2.png")
    
    # --- Part 2: Hierarchical Clustering (Dendrogram) ---
    # Use a smaller subset for readable dendrogram
    X_small = X[:50] 
    plot_dendrogram_chart(X_small, "clustering_dendrogram.png")

if __name__ == "__main__":
    main()
