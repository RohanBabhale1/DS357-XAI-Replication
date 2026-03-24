"""
run_spray.py
SpRAy pipeline — replicates paper Listing 2 algorithm using scipy + sklearn.
CoRelAy's SpectralClustering wraps these same libraries internally; we call
them directly to avoid a Param-slot API incompatibility with small N.

Run from project root: python spray_analysis/run_spray.py

Reads:  results/spray/heatmaps.npy
Writes: results/spray/spray.h5             (hashed HDF5 cache — CoRelAy format)
        results/spray/tsne_embedding.npy   (shape [N, 2])
        results/spray/cluster_labels.npy   (shape [N])
        results/spray/tsne_cluster_plot.png
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

np.random.seed(42)

# ── 1. Load and preprocess heatmaps ──────────────────────────────────────────
print("Loading heatmaps...")
data = np.load('results/spray/heatmaps.npy')          # (N, 3, 224, 224)
print(f"Loaded shape: {data.shape}")

# Sum channels → (N, 224, 224), L1-normalize, flatten → (N, H*W)
data_proc = data.sum(axis=1)
data_proc = data_proc / (np.abs(data_proc).sum(axis=(1, 2), keepdims=True) + 1e-9)
data_proc = data_proc.reshape(len(data_proc), -1).astype(np.float32)
print(f"Preprocessed shape: {data_proc.shape}")

N = len(data_proc)

# ── 2. Spectral embedding — same as CoRelAy EigenDecomposition ───────────────
print("Computing spectral embedding (EigenDecomposition)...")

# Build affinity matrix (cosine-like dot product, then normalize to [0,1])
affinity = data_proc @ data_proc.T                        # (N, N)
affinity = (affinity - affinity.min()) / (affinity.max() - affinity.min() + 1e-9)

n_eigval = min(8, N - 1)                                  # must be < N
eigenvalues, eigenvectors = eigsh(affinity, k=n_eigval, which='LM')
eigenvalues = 1.0 - eigenvalues                           # CoRelAy convention
eigenvectors = eigenvectors / (np.linalg.norm(eigenvectors, axis=1, keepdims=True) + 1e-9)
print(f"Eigenvectors shape: {eigenvectors.shape}")

# ── 3. t-SNE embedding — same as CoRelAy TSNEEmbedding ───────────────────────
print("Computing t-SNE embedding...")
perplexity = min(5, N - 1)                                # perplexity must be < N
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
tsne_embedding = tsne.fit_transform(eigenvectors)         # (N, 2)
print(f"t-SNE embedding shape: {tsne_embedding.shape}")

# ── 4. KMeans clustering ──────────────────────────────────────────────────────
chosen_k = min(3, N - 1)
print(f"Running KMeans with k={chosen_k}...")
kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(eigenvectors)         # (N,)
print(f"Cluster labels: {cluster_labels}")
print(f"Unique clusters: {np.unique(cluster_labels)}")

# ── 5. Save results ───────────────────────────────────────────────────────────
np.save('results/spray/tsne_embedding.npy', tsne_embedding)
np.save('results/spray/cluster_labels.npy', cluster_labels)
print("Saved tsne_embedding.npy and cluster_labels.npy")

# Also write into spray.h5 under proc_data so the file is populated
with h5py.File('results/spray/spray.h5', 'a') as f:
    grp = f.require_group('proc_data')
    # Overwrite if already exists
    for key in ['tsne_embedding', 'cluster_labels', 'eigenvectors']:
        if key in grp:
            del grp[key]
    grp.create_dataset('tsne_embedding',  data=tsne_embedding.astype(np.float32))
    grp.create_dataset('cluster_labels',  data=cluster_labels.astype(np.int64))
    grp.create_dataset('eigenvectors',    data=eigenvectors.astype(np.float32))
print("Updated spray.h5")

# ── 6. Plot t-SNE scatter ─────────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    tsne_embedding[:, 0], tsne_embedding[:, 1],
    c=cluster_labels, cmap='tab10', s=150, alpha=0.9
)
plt.title(f'SpRAy t-SNE embedding — {chosen_k} clusters (Lighthouse images)')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig('results/spray/tsne_cluster_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved tsne_cluster_plot.png")

print("\nDone!")
print("Next: python prepare_virelay.py")