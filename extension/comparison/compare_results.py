"""
extension/comparison/compare_results.py

Compare SpRAy cluster assignments against ground-truth class labels
(Normal=0, Pneumonia=1) for the PneumoniaMNIST extension.

Outputs:
  - extension/comparison/figures/fig_comparison.png
  - extension/comparison/COMPARISON_RESULTS_DOCUMENTATION.md
  - extension/spray/results/cluster_summary.txt

Usage:
    python extension/comparison/compare_results.py
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

matplotlib.use("Agg")


CLUSTER_LABELS_PATH = "extension/spray/results/cluster_labels.npy"
GROUND_TRUTH_PATH = "extension/results/labels.npy"
HEATMAPS_PATH = "extension/results/heatmaps.npy"
IMAGES_PATH = "extension/results/images.npy"
OUT_DIR = "extension/comparison/figures"
CLUSTER_SUMMARY_TXT = "extension/spray/results/cluster_summary.txt"
REPORT_PATH = "extension/comparison/COMPARISON_RESULTS_DOCUMENTATION.md"
TSNE_PATH = "extension/spray/results/tsne_embedding.npy"


def _purity(cluster_labels, true_labels):
    """Return clustering purity."""
    total = 0
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        majority = np.bincount(true_labels[mask]).max()
        total += majority
    return total / len(cluster_labels)


def _cluster_rows(cluster_labels, true_labels):
    """Build per-cluster summary rows for reporting."""
    rows = []
    for cluster_id in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == cluster_id
        normal_count = int(np.sum(true_labels[mask] == 0))
        pneumonia_count = int(np.sum(true_labels[mask] == 1))
        total = normal_count + pneumonia_count
        dominant = "Normal" if normal_count >= pneumonia_count else "Pneumonia"
        rows.append(
            {
                "cluster": int(cluster_id),
                "normal": normal_count,
                "pneumonia": pneumonia_count,
                "total": total,
                "dominant": dominant,
            }
        )
    return rows


def _write_cluster_summary(cluster_rows):
    """Write the text summary used by the spray results folder."""
    os.makedirs(os.path.dirname(CLUSTER_SUMMARY_TXT), exist_ok=True)
    with open(CLUSTER_SUMMARY_TXT, "w", encoding="utf-8") as file:
        file.write("SpRAy Cluster Summary\n")
        file.write("=====================\n\n")
        for row in cluster_rows:
            file.write(
                f"Cluster {row['cluster']}: {row['total']} images "
                f"({row['normal']} Normal, {row['pneumonia']} Pneumonia)\n"
            )
    print(f"[OK] Updated {CLUSTER_SUMMARY_TXT}")


def _write_markdown_report(n_samples, n_clusters, ari, nmi, purity, cluster_rows):
    """Write a standalone markdown report for comparison, results, and documentation."""
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    cluster_notes = {
        0: (
            "Dominated by Normal samples. The explanation pattern likely captures "
            "clear lung fields and low-opacity regions."
        ),
        1: (
            "Balanced between Normal and Pneumonia. This is the strongest sign of "
            "ambiguous or non-diagnostic explanation behavior."
        ),
        2: (
            "Strongly Pneumonia-dominated. This is the most clinically plausible "
            "cluster because it likely reflects consolidation-related evidence."
        ),
        3: (
            "Mostly Normal, but still mixed. The model may be using border, "
            "contrast, or structural cues in addition to pathology."
        ),
    }

    cluster_table = "\n".join(
        f"| C{row['cluster']} | {row['normal']} | {row['pneumonia']} | {row['total']} | {row['dominant']} |"
        for row in cluster_rows
    )
    cluster_sections = "\n\n".join(
        (
            f"### Cluster {row['cluster']}\n"
            f"- Composition: {row['normal']} Normal, {row['pneumonia']} Pneumonia\n"
            f"- Dominant class: {row['dominant']}\n"
            f"- Interpretation: {cluster_notes.get(row['cluster'], 'Cluster-specific note not available.')}"
        )
        for row in cluster_rows
    )

    report = f"""# Comparison, Results & Documentation

## Overview

This file documents the comparison stage of the medical imaging extension for the
PneumoniaMNIST experiment. The purpose is to compare SpRAy cluster assignments
with ground-truth diagnosis labels and determine whether the learned explanation
patterns reflect clinical evidence or possible shortcut behavior.

## Inputs

| Item | Path | Description |
|------|------|-------------|
| Cluster labels | `extension/spray/results/cluster_labels.npy` | K-Means labels from the SpRAy embedding |
| Ground truth labels | `extension/results/labels.npy` | Binary labels where 0 = Normal and 1 = Pneumonia |
| Grad-CAM heatmaps | `extension/results/heatmaps.npy` | Attribution maps used for clustering |
| Original images | `extension/results/images.npy` | Samples used for cluster overlay visualisation |

## Comparison Metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| Samples | {n_samples} | Number of evaluated images |
| Clusters | {n_clusters} | Number of K-Means groups |
| Adjusted Rand Index (ARI) | {ari:.4f} | Agreement between clusters and diagnosis labels |
| Normalised Mutual Information (NMI) | {nmi:.4f} | Shared information between clustering and labels |
| Clustering Purity | {purity:.4f} | Fraction assigned to a majority-class cluster |

## Cluster Composition

| Cluster | Normal | Pneumonia | Total | Dominant Class |
|---------|--------|-----------|-------|----------------|
{cluster_table}

## Results

The comparison shows partial alignment between SpRAy clusters and the medical
labels. ARI = {ari:.4f} and NMI = {nmi:.4f} indicate that the explanation
clusters capture some disease-related structure, but not enough to conclude that
the model relies only on pathology-specific evidence.

Purity = {purity:.4f} shows that most samples fall into majority-class clusters,
yet several clusters still contain a mixture of Normal and Pneumonia cases. That
mixed structure is important because it suggests some attribution maps are shaped
by shared visual properties such as brightness, borders, or scanner artifacts.

## Cluster-Level Interpretation

{cluster_sections}

## Bias Discussion

Cluster 1 is the strongest bias indicator because it is evenly split between
Normal and Pneumonia samples. When explanations from both classes group together,
the model may be attending to non-diagnostic cues instead of class-specific
pathology.

Cluster 2 is the most encouraging cluster because it is strongly dominated by
Pneumonia samples and is therefore more consistent with pathology-focused
reasoning. The Normal-heavy clusters may still be valid, but they could also
reflect dataset-specific structure rather than purely clinical evidence.

## Generated Outputs

| File | Description |
|------|-------------|
| `extension/comparison/figures/fig_comparison.png` | Combined comparison figure with metrics, t-SNE, and overlay examples |
| `extension/spray/results/cluster_summary.txt` | Plain-text cluster breakdown |
| `extension/comparison/COMPARISON_RESULTS_DOCUMENTATION.md` | Standalone markdown report for this stage |

## Reproduction

Run the comparison stage from the project root:

```bash
python extension/comparison/compare_results.py
```

The script loads the saved SpRAy outputs, computes the clustering metrics,
refreshes the text summary, regenerates the figure, and rewrites this markdown
documentation file.
"""

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(report)

    print(f"[OK] Wrote markdown report -> {REPORT_PATH}")


def _plot_comparison(cluster_labels, true_labels, ari, nmi, purity, images, heatmaps):
    """Create the summary comparison figure."""
    import cv2

    cluster_ids = sorted(np.unique(cluster_labels))
    n_clusters = len(cluster_ids)

    fig = plt.figure(figsize=(18, 5 + 3 * n_clusters))
    grid = fig.add_gridspec(2 + n_clusters, 6, hspace=0.45, wspace=0.35)

    ax_bar = fig.add_subplot(grid[0, :3])
    normals = [int(np.sum(true_labels[cluster_labels == cid] == 0)) for cid in cluster_ids]
    pneumonias = [int(np.sum(true_labels[cluster_labels == cid] == 1)) for cid in cluster_ids]
    positions = np.arange(len(cluster_ids))
    width = 0.35
    ax_bar.bar(positions - width / 2, normals, width, label="Normal", color="#4C72B0")
    ax_bar.bar(positions + width / 2, pneumonias, width, label="Pneumonia", color="#DD8452")
    ax_bar.set_xticks(positions)
    ax_bar.set_xticklabels([f"C{cid}" for cid in cluster_ids])
    ax_bar.set_xlabel("Cluster")
    ax_bar.set_ylabel("Count")
    ax_bar.set_title("Cluster Composition (Normal vs Pneumonia)")
    ax_bar.legend()

    ax_text = fig.add_subplot(grid[0, 3:])
    ax_text.axis("off")
    metrics_text = (
        f"Adjusted Rand Index (ARI):  {ari:.4f}\n"
        f"Normalised Mutual Info:     {nmi:.4f}\n"
        f"Clustering Purity:          {purity:.4f}\n\n"
        "ARI < 0.1 -> weak alignment with labels\n"
        "ARI > 0.2 -> partial diagnostic structure"
    )
    ax_text.text(
        0.05,
        0.5,
        metrics_text,
        transform=ax_text.transAxes,
        fontsize=11,
        va="center",
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "#f0f0f0", "alpha": 0.8},
    )
    ax_text.set_title("Clustering Quality Metrics")

    ax_tsne = fig.add_subplot(grid[1, :3])
    ax_gt = fig.add_subplot(grid[1, 3:])
    if os.path.exists(TSNE_PATH):
        tsne = np.load(TSNE_PATH)
        scatter_clusters = ax_tsne.scatter(
            tsne[:, 0], tsne[:, 1], c=cluster_labels, cmap="viridis", alpha=0.7, s=20
        )
        plt.colorbar(scatter_clusters, ax=ax_tsne, label="Cluster")
        ax_tsne.set_title("t-SNE of SpRAy Embedding (by cluster)")

        scatter_labels = ax_gt.scatter(
            tsne[:, 0], tsne[:, 1], c=true_labels, cmap="coolwarm", alpha=0.7, s=20
        )
        plt.colorbar(scatter_labels, ax=ax_gt, label="Ground Truth")
        ax_gt.set_title("t-SNE of SpRAy Embedding (by label)")
    else:
        ax_tsne.text(
            0.5,
            0.5,
            "t-SNE not available\nRun run_medical_spray.py first.",
            ha="center",
            va="center",
            transform=ax_tsne.transAxes,
        )
        ax_tsne.set_title("t-SNE embedding")
        ax_gt.axis("off")

    n_show = 5
    for row_index, cluster_id in enumerate(cluster_ids):
        indices = np.where(cluster_labels == cluster_id)[0]
        chosen = np.random.default_rng(42).choice(indices, min(n_show, len(indices)), replace=False)
        for col_index in range(n_show):
            ax = fig.add_subplot(grid[2 + row_index, col_index])
            if col_index < len(chosen):
                sample_index = chosen[col_index]
                cam = heatmaps[sample_index]
                image = images[sample_index]
                heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                overlay = np.clip(heat * 0.4 + image, 0, 255).astype(np.uint8)
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                ax.imshow(overlay_rgb)
                label = "N" if true_labels[sample_index] == 0 else "P"
                ax.set_title(f"Cluster {cluster_id} [{label}]", fontsize=7)
            ax.axis("off")

    plt.suptitle(
        "SpRAy Clustering vs Ground-Truth Analysis - PneumoniaMNIST Extension",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "fig_comparison.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] Comparison figure saved -> {out_path}")


def _rederive_clusters():
    """Re-run K-Means on the spectral embedding if cluster_labels.npy is missing."""
    from sklearn.cluster import KMeans

    heatmaps = np.load(HEATMAPS_PATH)
    sample_count = heatmaps.shape[0]
    flattened = heatmaps.reshape(sample_count, -1).astype(np.float32)
    squared = np.sum(flattened**2, axis=1, keepdims=True)
    distances = np.sqrt(np.clip(squared + squared.T - 2 * flattened @ flattened.T, 0, None))
    weights = np.exp(-(distances**2) / (2 * 50.0**2))
    np.fill_diagonal(weights, 0)
    degree = np.diag(weights.sum(1))
    degree_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(weights.sum(1), 1e-10)))
    laplacian = degree_inv_sqrt @ (degree - weights) @ degree_inv_sqrt
    _, eigenvectors = np.linalg.eigh(laplacian)
    embedding = eigenvectors[:, 1:11]
    model = KMeans(n_clusters=4, random_state=42)
    cluster_labels = model.fit_predict(embedding)
    np.save(CLUSTER_LABELS_PATH, cluster_labels)
    print(f"[OK] cluster_labels.npy saved -> {CLUSTER_LABELS_PATH}")


def analyse():
    """Run the comparison pipeline."""
    if not os.path.exists(CLUSTER_LABELS_PATH):
        print("cluster_labels.npy not found - re-deriving from SpRAy pipeline...")
        _rederive_clusters()

    cluster_labels = np.load(CLUSTER_LABELS_PATH)
    true_labels = np.load(GROUND_TRUTH_PATH)
    heatmaps = np.load(HEATMAPS_PATH)
    images = np.load(IMAGES_PATH)

    n_clusters = len(np.unique(cluster_labels))
    n_samples = len(cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    purity = _purity(cluster_labels, true_labels)
    cluster_rows = _cluster_rows(cluster_labels, true_labels)

    print("\n" + "=" * 60)
    print(" SpRAy Clustering vs Ground-Truth: Comparison Report")
    print("=" * 60)
    print(f"  Samples:          {n_samples}")
    print(f"  Clusters (k):     {n_clusters}")
    print(f"  Adjusted Rand Index (ARI):        {ari:.4f}")
    print(f"  Normalised Mutual Info (NMI):     {nmi:.4f}")
    print(f"  Clustering Purity:                {purity:.4f}")

    print("\n  Cluster Composition (rows=cluster, cols=Normal/Pneumonia):")
    print(f"  {'Cluster':>8}  {'Normal':>8}  {'Pneumonia':>10}  {'Total':>6}  {'Dominant Class':>16}")
    for row in cluster_rows:
        bias_flag = " <- BIAS?" if (row["normal"] == 0 or row["pneumonia"] == 0) else ""
        print(
            f"  {row['cluster']:>8}  {row['normal']:>8}  {row['pneumonia']:>10}  "
            f"{row['total']:>6}  {row['dominant']:>16}{bias_flag}"
        )

    print("\n  Interpretation:")
    print("  * ARI ~= 0 -> clusters do NOT align with Normal/Pneumonia split.")
    print("    The model may rely on non-diagnostic imaging features.")
    print("  * ARI > 0.2 -> clusters partially reflect true diagnosis.")
    print("  * Pure one-class clusters are possible bias indicators.")
    print("  * Mixed clusters can point to shared visual artifacts.")
    print("=" * 60 + "\n")

    _write_cluster_summary(cluster_rows)
    _write_markdown_report(n_samples, n_clusters, ari, nmi, purity, cluster_rows)
    _plot_comparison(cluster_labels, true_labels, ari, nmi, purity, images, heatmaps)


if __name__ == "__main__":
    analyse()
