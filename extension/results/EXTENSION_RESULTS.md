# Extension Results: SpRAy Clustering & Bias Detection on Medical Imaging

**Task:** Member D — Apply SpRAy (Spectral Relevance Analysis) to a medical imaging domain (chest X-ray pneumonia detection) and detect potential model biases.

---

## 1. Setup

| Item | Detail |
|------|--------|
| **Dataset** | PneumoniaMNIST (MedMNIST v2, Yang et al. 2023 — Kermany et al. 2018 chest X-rays) |
| **Subset** | 100 Normal + 100 Pneumonia images (200 total, `split='train'`) |
| **Model** | VGG-16-BN pretrained on ImageNet, fine-tuned for binary classification |
| **XAI Method** | Grad-CAM (pixel-level attribution via gradient × activation) |
| **Clustering** | Spectral clustering (Graph Laplacian) → K-Means (k=4) → t-SNE visualization |

---

## 2. Model Performance

Training for 5 epochs, frozen convolutional backbone, only classifier head trained.

| Epoch | Val Acc | Val F1 | Val AUC |
|-------|---------|--------|---------|
| 1 | 0.850 | 0.875 | 0.922 |
| 2 | 0.650 | 0.750 | **0.962** ← best |
| 3 | 0.750 | 0.800 | 0.925 |
| 4 | 0.800 | 0.826 | 0.945 |
| 5 | 0.875 | 0.884 | 0.960 |

Best model checkpoint saved at **val_AUC = 0.962** (Epoch 2).

---

## 3. SpRAy Clustering Results

### 3.1 Cluster Composition

| Cluster | Normal | Pneumonia | Total | Dominant Class |
|---------|--------|-----------|-------|----------------|
| **C0** | 36 | 3 | 39 | Normal |
| **C1** | 22 | 22 | 44 | Mixed |
| **C2** | 7 | 62 | 69 | **Pneumonia** |
| **C3** | 35 | 13 | 48 | Normal |

### 3.2 Quantitative Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Adjusted Rand Index (ARI) | **0.2242** | Partial alignment with diagnosis labels |
| Normalised Mutual Info (NMI) | **0.2281** | Moderate information overlap with GT |
| Clustering Purity | **0.775** | 77.5% of samples assigned to majority-class cluster |

---

## 4. Bias Detection Analysis

### Key Finding: ARI = 0.22 (Partial Alignment)

An ARI of 0.22 means the Grad-CAM heatmap clusters **partially but not fully** map onto the Normal/Pneumonia split. This is consistent with SpRAy findings in the original paper (Bach et al., Lapuschkin et al.) where features beyond the intended diagnostic signal influence model decisions.

### Cluster Interpretations

**Cluster 0 (36 Normal, 3 Pneumonia) — "Clear Lung" strategy:**
- Dominated by normal images. Grad-CAM focuses on uniform, clear lung fields.
- Possible bias: model may associate _absence_ of pattern with healthy label, rather than specific anatomical features.

**Cluster 1 (22 Normal, 22 Pneumonia) — "Mixed / Ambiguous" attribution:**
- Perfectly balanced between classes. The Grad-CAM heatmaps in this cluster are likely diffuse or attend to non-discriminative regions (background, edges, image frame).
- **Bias signal:** Model attention is not class-informative — consistent with attention to imaging artifacts or intensity gradients rather than pathological changes.

**Cluster 2 (7 Normal, 62 Pneumonia) — "Pneumonia Consolidation" strategy:**
- Strongly pneumonia-dominated. Grad-CAM likely highlights high-opacity consolidation regions in the lower lobes.
- This is the most diagnostically interpretable cluster.

**Cluster 3 (35 Normal, 13 Pneumonia) — "Peripheral/Border" strategy:**
- Mostly normal images. Grad-CAM may be attending to the lung border or diaphragm boundary.
- Could reflect a dataset bias where normal lungs are sharply bordered vs. pneumonia cases with blurred margins.

### Conclusion

The SpRAy analysis reveals that ~22% of the clustering structure overlaps with the clinical Normal/Pneumonia distinction (ARI = 0.22). **Cluster 1 is the strongest bias indicator**, as balanced class membership with Grad-CAM suggests the model is attending to non-diagnostic features for a significant subset of predictions. This matches prior findings that deep neural networks trained on small medical imaging datasets frequently exploit dataset-specific visual artifacts (image borders, brightness, scanner-related patterns) instead of purely clinical signals.

---

## 5. Output Files

| File | Description |
|------|-------------|
| `extension/models/vgg16_chest.pth` | Fine-tuned VGG-16-BN weights |
| `extension/results/heatmaps.npy` | Grad-CAM heatmaps (200, 224, 224) |
| `extension/results/labels.npy` | Ground-truth labels (200,) |
| `extension/results/images.npy` | Original images (200, 224, 224, 3) |
| `extension/spray/results/cluster_labels.npy` | K-Means cluster assignments (200,) |
| `extension/spray/results/tsne_embedding.npy` | t-SNE 2D embedding (200, 2) |
| `extension/spray/results/fig_spray_clusters.png` | t-SNE cluster plot |
| `extension/spray/results/fig_cluster_heatmaps.png` | Per-cluster Grad-CAM overlays |
| `extension/comparison/figures/fig_comparison.png` | Full comparison figure |

---

## 6. Scripts

| Script | Purpose |
|--------|---------|
| `extension/data/download_chest_data.py` | Download & save PneumoniaMNIST subset |
| `extension/models/train_chest_model.py` | Fine-tune VGG-16-BN |
| `extension/heatmaps/generate_batch_heatmaps.py` | Batch Grad-CAM heatmap generation |
| `extension/spray/run_medical_spray.py` | SpRAy spectral clustering + t-SNE |
| `extension/comparison/compare_results.py` | ARI/NMI bias report + comparison figure |

---

## 7. References

- Bach et al. (2015) — *On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation*, PLOS ONE.
- Lapuschkin et al. (2019) — *Unmasking Clever Hans predictors and assessing what machines really learn*, Nature Communications.
- Selvaraju et al. (2017) — *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV.
- Yang et al. (2023) — *MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification*, Scientific Data.
- Kermany et al. (2018) — *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell.
