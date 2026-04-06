# DS357-XAI-Replication

XAI course project: Replicating Layer-wise Relevance Propagation (LRP) attribution methods

A Python implementation for computing pixel-level attribution heatmaps using Layer-wise Relevance Propagation (LRP) and related methods via the Zennit library. Includes utilities for generating heatmaps, SpRAy clustering analysis, and visualization.

## File Structure

```
DS357-XAI-Replication/
├── README.md                          # Project documentation
├── LICENSE                            # License file
├── requirements.txt                   # Python dependencies
│
├── smoke_test.py                      # Quick sanity check for Zennit setup
├── test_single_heatmap.py             # Example: compute LRP for a single image
│
├── data/
│   └── imagenet_samples/              # Sample ImageNet images (lighthouse class)
│
├── results/
│   ├── heatmaps/                      # Generated attribution heatmaps
│   └── spray/                         # SpRAy clustering results
│       ├── cluster_labels.npy         # Cluster assignments (shape [N])
│       ├── heatmaps.npy               # Input heatmaps (shape [N, 3, 224, 224])
│       ├── spray.h5                   # HDF5 cache (CoRelAy format)
│       └── tsne_embedding.npy         # t-SNE embedding (shape [N, 2])
│
├── xai_methods/
│   ├── __init__.py
│   └── lrp_zennit.py                  # Core LRP implementation using Zennit
│
└── spray_analysis/
    ├── generate_heatmaps.py           # Generate heatmaps for multiple images & methods
    └── run_spray.py                   # SpRAy clustering pipeline
```

## How to Use

### 1. Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Quick Test

Verify your Zennit installation works correctly:
```bash
python smoke_test.py
```

This tests VGG-16-BN with the EpsilonGammaBox composite rule.

### 3. Generate Heatmap for a Single Image

Compute LRP attribution for one image (lighthouse_001.jpg):
```bash
python test_single_heatmap.py
```

This:
- Loads the test image and pre-trained VGG-16-BN model
- Computes LRP using the EpsilonGammaBox composite
- Targets ImageNet class 437 (lighthouse)
- Saves the heatmap to `results/heatmaps/test_single.png`

### 4. Generate Heatmaps for Multiple Images and Methods

Create heatmaps for all images using multiple attribution methods:
```bash
python spray_analysis/generate_heatmaps.py
```

This script:
- Processes all JPEG images in `data/imagenet_samples/`
- Applies three LRP composites: EpsilonGammaBox, EpsilonPlus, EpsilonAlpha2Beta1
- Generates a comparison grid visualization
- Saves results to `results/heatmaps/`

### 5. SpRAy Clustering Analysis

Run the SpRAy (Spectral Relevance Analysis) pipeline on generated heatmaps:
```bash
python spray_analysis/run_spray.py
```

This script:
- Loads preprocessed heatmaps from `results/spray/heatmaps.npy`
- Computes a spectral embedding (similar to CoRelAy's EigenDecomposition)
- Performs K-Means clustering
- Generates t-SNE visualization
- Outputs:
  - `results/spray/spray.h5` — HDF5 cache in CoRelAy format
  - `results/spray/tsne_embedding.npy` — t-SNE coordinates
  - `results/spray/cluster_labels.npy` — Cluster assignments
  - `results/spray/tsne_cluster_plot.png` — Visualization

## Key Components :-

### xai_methods/lrp_zennit.py

Main module for LRP computation:
- `get_composite(composite_name, model_name)` — Get Zennit composite rule (supports VGG & ResNet)
- `compute_lrp(model, image_tensor, composite_name, model_name, target_class)` — Compute attribution heatmap
- `heatmap_to_image(relevance)` — Convert relevance tensor to PIL Image

**Supported Composites:**
- `EpsilonGammaBox` — Recommended composite (paper default)
- `EpsilonPlus` — Adding perturbation
- `EpsilonAlpha2Beta1` — Layer-wise α-β relevance propagation

## Dependencies

- **torch**, **torchvision** — Deep learning framework
- **zennit** — Layer-wise Relevance Propagation implementation
- **numpy**, **scipy**, **scikit-learn** — Scientific computing
- **h5py** — HDF5 file handling
- **matplotlib**, **pillow** — Visualization
- **tqdm** — Progress bars
- **jupyter**, **pandas** — Data analysis 
