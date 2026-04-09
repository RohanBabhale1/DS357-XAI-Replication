# Comparison, Results & Documentation

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
| Samples | 200 | Number of evaluated images |
| Clusters | 4 | Number of K-Means groups |
| Adjusted Rand Index (ARI) | 0.1532 | Agreement between clusters and diagnosis labels |
| Normalised Mutual Information (NMI) | 0.2044 | Shared information between clustering and labels |
| Clustering Purity | 0.7350 | Fraction assigned to a majority-class cluster |

## Cluster Composition

| Cluster | Normal | Pneumonia | Total | Dominant Class |
|---------|--------|-----------|-------|----------------|
| C0 | 31 | 5 | 36 | Normal |
| C1 | 38 | 21 | 59 | Normal |
| C2 | 30 | 26 | 56 | Normal |
| C3 | 1 | 48 | 49 | Pneumonia |

## Results

The comparison shows partial alignment between SpRAy clusters and the medical
labels. ARI = 0.1532 and NMI = 0.2044 indicate that the explanation
clusters capture some disease-related structure, but not enough to conclude that
the model relies only on pathology-specific evidence.

Purity = 0.7350 shows that most samples fall into majority-class clusters,
yet several clusters still contain a mixture of Normal and Pneumonia cases. That
mixed structure is important because it suggests some attribution maps are shaped
by shared visual properties such as brightness, borders, or scanner artifacts.

## Cluster-Level Interpretation

### Cluster 0
- Composition: 31 Normal, 5 Pneumonia
- Dominant class: Normal
- Interpretation: Dominated by Normal samples. The explanation pattern likely captures clear lung fields and low-opacity regions.

### Cluster 1
- Composition: 38 Normal, 21 Pneumonia
- Dominant class: Normal
- Interpretation: Balanced between Normal and Pneumonia. This is the strongest sign of ambiguous or non-diagnostic explanation behavior.

### Cluster 2
- Composition: 30 Normal, 26 Pneumonia
- Dominant class: Normal
- Interpretation: Strongly Pneumonia-dominated. This is the most clinically plausible cluster because it likely reflects consolidation-related evidence.

### Cluster 3
- Composition: 1 Normal, 48 Pneumonia
- Dominant class: Pneumonia
- Interpretation: Mostly Normal, but still mixed. The model may be using border, contrast, or structural cues in addition to pathology.

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
