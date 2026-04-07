"""
extension/comparison/compare_results.py

Compares SpRAy cluster assignments against ground-truth class labels (Normal=0,
Pneumonia=1) for the PneumoniaMNIST extension.  Produces:
  - extension/comparison/figures/fig_comparison.png  (already present from prior run)
  - Prints a detailed bias-detection report to stdout

Usage:
    python extension/comparison/compare_results.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
)

# ------------------------------------------------------------------ paths ---
CLUSTER_LABELS_PATH = "extension/spray/results/cluster_labels.npy"
GROUND_TRUTH_PATH   = "extension/results/labels.npy"
HEATMAPS_PATH       = "extension/results/heatmaps.npy"
IMAGES_PATH         = "extension/results/images.npy"
OUT_DIR             = "extension/comparison/figures"
CLUSTER_SUMMARY_TXT = "extension/spray/results/cluster_summary.txt"


# -------------------------------------------------------- helper functions ---
def _purity(cluster_labels, true_labels):
    """Clustering purity: fraction of samples that match majority class."""
    n = len(cluster_labels)
    total = 0
    for cid in np.unique(cluster_labels):
        mask = cluster_labels == cid
        majority = np.bincount(true_labels[mask]).max()
        total += majority
    return total / n


def _write_cluster_summary(cluster_labels, true_labels):
    """Write / overwrite the cluster summary text file."""
    os.makedirs(os.path.dirname(CLUSTER_SUMMARY_TXT), exist_ok=True)
    with open(CLUSTER_SUMMARY_TXT, "w") as f:
        f.write("SpRAy Cluster Summary\n")
        f.write("=====================\n\n")
        for cid in sorted(np.unique(cluster_labels)):
            mask = cluster_labels == cid
            n_normal    = int(np.sum(true_labels[mask] == 0))
            n_pneumonia = int(np.sum(true_labels[mask] == 1))
            f.write(f"Cluster {cid}: {int(mask.sum())} images "
                    f"({n_normal} Normal, {n_pneumonia} Pneumonia)\n")
    print(f"\n✓ Updated {CLUSTER_SUMMARY_TXT}")


# --------------------------------------------------------------- analysis ---
def analyse():
    # -- load --
    if not os.path.exists(CLUSTER_LABELS_PATH):
        # Derive cluster labels from heatmaps if the npy was not saved separately
        # (run_medical_spray.py saves cluster labels in memory only in this version)
        print("cluster_labels.npy not found – re-deriving from SpRAy pipeline …")
        _rederive_clusters()

    cluster_labels = np.load(CLUSTER_LABELS_PATH)
    true_labels    = np.load(GROUND_TRUTH_PATH)
    heatmaps       = np.load(HEATMAPS_PATH)
    images         = np.load(IMAGES_PATH)

    n_clusters = len(np.unique(cluster_labels))
    n_samples  = len(cluster_labels)

    # -- metrics --
    ari   = adjusted_rand_score(true_labels, cluster_labels)
    nmi   = normalized_mutual_info_score(true_labels, cluster_labels)
    purity = _purity(cluster_labels, true_labels)

    print("\n" + "="*60)
    print(" SpRAy Clustering vs Ground-Truth: Comparison Report")
    print("="*60)
    print(f"  Samples:          {n_samples}")
    print(f"  Clusters (k):     {n_clusters}")
    print(f"  Adjusted Rand Index (ARI):        {ari:.4f}")
    print(f"  Normalised Mutual Info (NMI):     {nmi:.4f}")
    print(f"  Clustering Purity:                {purity:.4f}")

    print("\n  Cluster Composition (rows=cluster, cols=Normal/Pneumonia):")
    print(f"  {'Cluster':>8}  {'Normal':>8}  {'Pneumonia':>10}  {'Total':>6}  {'Dominant Class':>16}")
    for cid in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == cid
        n  = int(np.sum(true_labels[mask] == 0))
        p  = int(np.sum(true_labels[mask] == 1))
        tot = n + p
        dom = "Normal" if n >= p else "Pneumonia"
        bias_flag = " ◄ BIAS?" if (n == 0 or p == 0) else ""
        print(f"  {cid:>8}  {n:>8}  {p:>10}  {tot:>6}  {dom:>16}{bias_flag}")

    print("\n  Interpretation:")
    print("  · ARI ≈ 0 → clusters do NOT align with Normal/Pneumonia split")
    print("    (model may rely on non-diagnostic imaging features).")
    print("  · ARI > 0.2 → clusters partially reflect true diagnosis.")
    print("  · Any cluster with 0 of one class is a potential BIAS cluster.")
    print("  · Clusters dominated by one class but drawn from both represent")
    print("    shared visual artifacts (e.g. lung texture, brightness gradients).")
    print("="*60 + "\n")

    # -- update cluster summary --
    _write_cluster_summary(cluster_labels, true_labels)

    # -- visualisation --
    os.makedirs(OUT_DIR, exist_ok=True)
    _plot_comparison(cluster_labels, true_labels, ari, nmi, purity, images, heatmaps)


def _plot_comparison(cluster_labels, true_labels, ari, nmi, purity, images, heatmaps):
    import cv2

    n_clusters = len(np.unique(cluster_labels))

    fig = plt.figure(figsize=(18, 5 + 3 * n_clusters))
    gs  = fig.add_gridspec(
        2 + n_clusters, 6,
        hspace=0.45, wspace=0.35,
    )

    # ---- Row 0: bar chart of cluster composition ----
    ax_bar = fig.add_subplot(gs[0, :3])
    ids       = sorted(np.unique(cluster_labels))
    normals   = [int(np.sum(true_labels[cluster_labels == c] == 0)) for c in ids]
    pneumonias = [int(np.sum(true_labels[cluster_labels == c] == 1)) for c in ids]
    x = np.arange(len(ids))
    width = 0.35
    ax_bar.bar(x - width/2, normals,    width, label="Normal",    color="#4C72B0")
    ax_bar.bar(x + width/2, pneumonias, width, label="Pneumonia", color="#DD8452")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"C{c}" for c in ids])
    ax_bar.set_xlabel("Cluster"); ax_bar.set_ylabel("Count")
    ax_bar.set_title("Cluster Composition (Normal vs Pneumonia)")
    ax_bar.legend()

    # ---- Row 0: metric summary text ----
    ax_txt = fig.add_subplot(gs[0, 3:])
    ax_txt.axis("off")
    metrics_txt = (
        f"Adjusted Rand Index (ARI):  {ari:.4f}\n"
        f"Normalised Mutual Info:     {nmi:.4f}\n"
        f"Clustering Purity:          {purity:.4f}\n\n"
        f"ARI < 0.1 → clusters NOT aligned with labels\n"
        f"ARI ≈ 0   → possible imaging artifact bias"
    )
    ax_txt.text(0.05, 0.5, metrics_txt, transform=ax_txt.transAxes,
                fontsize=11, va="center", family="monospace",
                bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
    ax_txt.set_title("Clustering Quality Metrics")

    # ---- Rows 1…n_clusters: heatmap overlays per cluster ----
    n_show = 5
    for row_i, cid in enumerate(ids):
        idxs = np.where(cluster_labels == cid)[0]
        chosen = np.random.default_rng(42).choice(idxs, min(n_show, len(idxs)), replace=False)
        for col_j in range(n_show):
            ax = fig.add_subplot(gs[2 + row_i, col_j])
            if col_j < len(chosen):
                idx = chosen[col_j]
                cam  = heatmaps[idx]
                img  = images[idx]
                heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                over = np.clip(heat * 0.4 + img, 0, 255).astype(np.uint8)
                over_rgb = cv2.cvtColor(over, cv2.COLOR_BGR2RGB)
                ax.imshow(over_rgb)
                lbl = "N" if true_labels[idx] == 0 else "P"
                ax.set_title(f"Cluster {cid} [{lbl}]", fontsize=7)
            ax.axis("off")
        # label the row
        fig.text(0.01, ax.get_position().y1, f"C{cid}", va="top", fontsize=10,
                 color="steelblue", fontweight="bold")

    # ---- Row 1: t-SNE plot if embedding exists ----
    tsne_path = "extension/spray/results/tsne_embedding.npy"
    ax_tsne = fig.add_subplot(gs[1, :3])
    if os.path.exists(tsne_path):
        tsne = np.load(tsne_path)
        sc = ax_tsne.scatter(tsne[:, 0], tsne[:, 1], c=cluster_labels,
                             cmap="viridis", alpha=0.7, s=20)
        plt.colorbar(sc, ax=ax_tsne, label="Cluster")
        ax_tsne.set_title("t-SNE of SpRAy Embedding (coloured by cluster)")
    else:
        ax_tsne.text(0.5, 0.5, "t-SNE not available\n(run run_medical_spray.py first)",
                     ha="center", va="center", transform=ax_tsne.transAxes)
        ax_tsne.set_title("t-SNE embedding")
    ax_tsne.axis("off")

    ax_gt = fig.add_subplot(gs[1, 3:])
    if os.path.exists(tsne_path):
        tsne = np.load(tsne_path)
        sc2 = ax_gt.scatter(tsne[:, 0], tsne[:, 1], c=true_labels,
                             cmap="coolwarm", alpha=0.7, s=20)
        plt.colorbar(sc2, ax=ax_gt, label="GT Label")
        ax_gt.set_title("t-SNE coloured by Ground-Truth Label")
    else:
        ax_gt.axis("off")
    ax_gt.axis("off")

    plt.suptitle("SpRAy Clustering vs Ground-Truth Analysis — PneumoniaMNIST Extension",
                 fontsize=13, fontweight="bold", y=1.01)
    out_path = os.path.join(OUT_DIR, "fig_comparison.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"✓ Comparison figure saved → {out_path}")


def _rederive_clusters():
    """Re-run K-Means on spectral embedding if cluster_labels.npy is missing."""
    from sklearn.cluster import KMeans
    heatmaps = np.load(HEATMAPS_PATH)
    N = heatmaps.shape[0]
    X = heatmaps.reshape(N, -1).astype(np.float32)
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    dist  = np.sqrt(np.clip(X_sq + X_sq.T - 2 * X @ X.T, 0, None))
    W = np.exp(-(dist**2) / (2 * 50.0**2)); np.fill_diagonal(W, 0)
    D = np.diag(W.sum(1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(W.sum(1), 1e-10)))
    L_sym = D_inv_sqrt @ (D - W) @ D_inv_sqrt
    _, vecs = np.linalg.eigh(L_sym)
    emb = vecs[:, 1:11]
    km  = KMeans(n_clusters=4, random_state=42).fit(emb)
    np.save(CLUSTER_LABELS_PATH, km.labels_)
    print(f"✓ cluster_labels.npy saved → {CLUSTER_LABELS_PATH}")


if __name__ == "__main__":
    analyse()
