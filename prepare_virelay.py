"""
Prepare ViRelAy HDF5 databases and project YAML file.
ViRelAy is a Flask web server (not a Python library) — it reads from these HDF5 files.

Run: python prepare_virelay.py
Then start ViRelAy: virelay results/virelay/project.yaml
Open browser: http://localhost:8080

Reference: https://virelay.rtfd.io/en/0.4.0/contributors-guide/database-specification.html
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import h5py
from PIL import Image
from torchvision import transforms

SPRAY_DIR = Path('results/spray')
VOC_DIR = Path('data/pascal_voc')
VIRELAY_DIR = Path('results/virelay')

CLASS_MAP = {0: 'bird', 1: 'horse'}
IMG_SIZE = 224
MAX_PER_CLASS = 60


def load_voc_images_array():
    """Load VOC images as numpy array in same order as heatmaps_voc.npy."""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    images, labels, paths = [], [], []
    for label_idx, class_name in CLASS_MAP.items():
        class_dir = VOC_DIR / class_name
        if not class_dir.exists():
            print(f"  ⚠️  {class_dir} not found, skipping")
            continue
        for p in sorted(class_dir.glob('*.jpg'))[:MAX_PER_CLASS]:
            try:
                img = Image.open(p).convert('RGB')
                t = transform(img).numpy()   # (3, H, W), values 0-1
                images.append(t)
                labels.append(label_idx)
                paths.append(str(p))
            except Exception as e:
                print(f"  ⚠️  {p.name}: {e}")
    return np.stack(images), np.array(labels), paths


def write_dataset_h5(images_np, labels_np, out_path):
    """Write dataset HDF5 for ViRelAy."""
    with h5py.File(str(out_path), 'w') as f:
        f.create_dataset('data', data=images_np, compression='gzip')
        f.create_dataset('labels', data=labels_np)
        # ViRelAy expects label names as bytes
        label_names = np.array([b'bird', b'horse'])
        f.create_dataset('label_names', data=label_names)
    print(f"  ✅ dataset.h5: {images_np.shape}")


def write_attribution_h5(heatmaps_np, out_path):
    """Write attribution HDF5 for ViRelAy."""
    with h5py.File(str(out_path), 'w') as f:
        f.create_dataset('attribution', data=heatmaps_np, compression='gzip')
    print(f"  ✅ attribution.h5: {heatmaps_np.shape}")


def write_analysis_h5(tsne_np, cluster_np, out_path):
    """Write analysis (SpRAy results) HDF5 for ViRelAy."""
    with h5py.File(str(out_path), 'w') as f:
        f.create_dataset('embedding', data=tsne_np)
        f.create_dataset('cluster_labels', data=cluster_np.astype(np.int32))
    print(f"  ✅ analysis.h5: tsne {tsne_np.shape}, clusters {cluster_np.shape}")


def write_project_yaml(dataset_h5, attribution_h5, analysis_h5, out_path):
    """Write ViRelAy project YAML file."""
    yaml_content = f"""# ViRelAy Project Configuration
# DS357 XAI Phase 2 — SpRAy Analysis (PASCAL VOC 2007)
# Reference: https://virelay.rtfd.io/en/0.4.0/contributors-guide/project-file-format.html

project_name: "DS357 XAI Phase 2 — Bird vs Horse SpRAy"

dataset:
  path: "{dataset_h5.resolve()}"
  data_key: "data"
  label_key: "labels"
  label_names_key: "label_names"

attribution:
  path: "{attribution_h5.resolve()}"
  attribution_key: "attribution"
  method: "LRP-EpsilonGammaBox (Zennit)"

analysis:
  path: "{analysis_h5.resolve()}"
  embedding_key: "embedding"
  cluster_label_key: "cluster_labels"
"""
    with open(str(out_path), 'w', encoding='utf-8', newline='\n') as f:
        f.write(yaml_content)
    print(f"  ✅ project.yaml written")


def main():
    print("=" * 60)
    print("DS357 Phase 2 — Prepare ViRelAy Files")
    print("=" * 60)

    VIRELAY_DIR.mkdir(parents=True, exist_ok=True)

    # Load images
    print("\nLoading VOC images...")
    images_np, labels_np, _ = load_voc_images_array()
    print(f"  Images: {images_np.shape}, Labels: {labels_np.shape}")

    # Load heatmaps
    heatmaps_file = SPRAY_DIR / 'heatmaps_voc.npy'
    if not heatmaps_file.exists():
        print(f"❌ {heatmaps_file} not found. Run compute_voc_heatmaps.py first.")
        return
    heatmaps_np = np.load(heatmaps_file)

    # Load SpRAy results
    tsne_file = SPRAY_DIR / 'tsne_embedding.npy'
    cluster_file = SPRAY_DIR / 'cluster_labels.npy'
    if not tsne_file.exists() or not cluster_file.exists():
        print(f"❌ SpRAy results not found. Run run_spray.py first.")
        return
    tsne_np = np.load(tsne_file)
    cluster_np = np.load(cluster_file)

    # Align sizes (images and heatmaps must match)
    n = min(len(images_np), len(heatmaps_np), len(tsne_np))
    images_np = images_np[:n]
    labels_np = labels_np[:n]
    heatmaps_np = heatmaps_np[:n]
    tsne_np = tsne_np[:n]
    cluster_np = cluster_np[:n]
    print(f"\nAligned to N={n} samples")

    # Write HDF5 files
    print("\nWriting ViRelAy HDF5 files...")
    dataset_h5 = VIRELAY_DIR / 'dataset.h5'
    attribution_h5 = VIRELAY_DIR / 'attribution.h5'
    analysis_h5 = VIRELAY_DIR / 'analysis.h5'

    write_dataset_h5(images_np, labels_np, dataset_h5)
    write_attribution_h5(heatmaps_np, attribution_h5)
    write_analysis_h5(tsne_np, cluster_np, analysis_h5)
    write_project_yaml(dataset_h5, attribution_h5, analysis_h5,
                       VIRELAY_DIR / 'project.yaml')

    print("\n✅ ViRelAy files ready!")
    print(f"\nTo start ViRelAy:")
    print(f"  virelay {VIRELAY_DIR / 'project.yaml'}")
    print(f"  Open: http://localhost:8080")
    print("\nTake a screenshot of the interface for your submission.")


if __name__ == '__main__':
    main()