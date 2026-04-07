import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import cv2

HEATMAPS_PATH = "extension/results/heatmaps.npy"
LABELS_PATH = "extension/results/labels.npy"
IMAGES_PATH = "extension/results/images.npy"
RESULTS_DIR = "extension/spray/results"

def load_data():
    heatmaps = np.load(HEATMAPS_PATH)
    labels = np.load(LABELS_PATH)
    images = np.load(IMAGES_PATH)
    return heatmaps, labels, images

def perform_spectral_embedding(heatmaps, sigma=0.5):
    # Flatten heatmaps (N, 224, 224) -> (N, 224*224)
    N = heatmaps.shape[0]
    X = heatmaps.reshape(N, -1)
    
    # Compute pairwise Euclidean distances
    print("Computing distance matrix...")
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    dist_matrix = X_sq + X_sq.T - 2 * np.dot(X, X.T)
    dist_matrix = np.clip(dist_matrix, 0, None) # avoid negative due to floating point
    dist_matrix = np.sqrt(dist_matrix)
    
    # Compute adjacency matrix with RBF kernel
    print("Computing adjacency matrix...")
    W = np.exp(- (dist_matrix**2) / (2 * sigma**2))
    np.fill_diagonal(W, 0)
    
    # Compute graph Laplacian
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    
    # Normalized Laplacian
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt
    
    # Eigen decomposition
    print("Computing eigenvectors...")
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
    
    # Use top k smallest eigenvectors (excluding the first one which is trivial)
    k = 10
    embedding = eigenvectors[:, 1:k+1]
    
    return embedding

def apply_heatmap(img, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = heatmap * 0.4 + img
    return overlay.astype(np.uint8)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Loading data...")
    heatmaps, labels, images = load_data()
    N = len(heatmaps)
    
    # 1. Spectral Embedding
    embedding = perform_spectral_embedding(heatmaps, sigma=50.0)
    
    # 2. K-Means Clustering on the embedding
    n_clusters = 4
    print(f"Running K-Means clustering with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embedding)
    np.save(os.path.join(RESULTS_DIR, 'cluster_labels.npy'), cluster_labels)
    
    # 3. t-SNE Visualization
    print("Running t-SNE...")
    tsne_embedding = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(embedding)
    np.save(os.path.join(RESULTS_DIR, 'tsne_embedding.npy'), tsne_embedding)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title('SpRAy t-SNE Visualization of Heatmaps')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(os.path.join(RESULTS_DIR, 'fig_spray_clusters.png'))
    plt.close()
    
    # 4. Generate Cluster Prototypes
    print("Generating cluster prototypes...")
    n_cols = 5
    fig, axes = plt.subplots(n_clusters, n_cols, figsize=(15, 3*n_clusters))
    
    with open(os.path.join(RESULTS_DIR, 'cluster_summary.txt'), 'w') as f:
        f.write("SpRAy Cluster Summary\n")
        f.write("=====================\n\n")
        
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            normal_count = np.sum(labels[cluster_indices] == 0)
            pneumonia_count = np.sum(labels[cluster_indices] == 1)
            
            f.write(f"Cluster {i}: {len(cluster_indices)} images "
                    f"({normal_count} Normal, {pneumonia_count} Pneumonia)\n")
            
            # Select 5 representative images for each cluster
            # (Here we just pick random ones for visualization)
            plot_indices = np.random.choice(cluster_indices, min(n_cols, len(cluster_indices)), replace=False)
            
            for j in range(n_cols):
                ax = axes[i, j]
                if j < len(plot_indices):
                    idx = plot_indices[j]
                    overlay = apply_heatmap(images[idx], heatmaps[idx])
                    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                    label_str = "Pneumonia" if labels[idx] == 1 else "Normal"
                    if j == 0:
                        ax.set_title(f"Cluster {i} ({label_str})")
                    else:
                        ax.set_title(label_str)
                ax.axis('off')
                
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig_cluster_heatmaps.png'))
    plt.close()
    
    print(f"SpRAy analysis complete. Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
