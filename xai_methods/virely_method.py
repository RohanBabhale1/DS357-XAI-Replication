"""
ViRelAy integration utilities for Phase 2 XAI replication.

IMPORTANT: ViRelAy is NOT a Python attribution library.
It is a Flask web server for interactive browsing of SpRAy cluster results.
There is no ViRelAy(model).explain() call — that API does not exist.

Correct usage:
    1. Generate LRP heatmaps with Zennit  (xai_methods/lrp_zennit.py)
    2. Run SpRAy clustering with CoRelAy  (spray_analysis/run_spray.py)
    3. Use this file to write HDF5 output files in the format ViRelAy expects
    4. Launch the ViRelAy web server from the command line:
           virelay path/to/project.yaml
    5. Open http://localhost:8080 in your browser

Reference: Paper Figure 2 (page 7) and Appendix A.1 (page 17)
HDF5 spec: https://virelay.rtfd.io/en/0.4.0/contributors-guide/database-specification.html
YAML spec:  https://virelay.rtfd.io/en/0.4.0/contributors-guide/project-file-format.html
"""

import os
import subprocess
import textwrap
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch


# ---------------------------------------------------------------------------
# HDF5 file writers
# ---------------------------------------------------------------------------

def write_dataset_hdf5(
    images: np.ndarray,
    labels: np.ndarray,
    output_path: str,
) -> None:
    """
    Write the dataset HDF5 file that ViRelAy requires.

    ViRelAy expects input images stored under the key 'data' with shape
    [N, C, H, W] (channels-first, float32, values in [0, 1]).

    Args:
        images (np.ndarray): Image array, shape [N, C, H, W], float32, [0, 1].
        labels (np.ndarray): Integer class labels, shape [N].
        output_path (str): Destination path, e.g. 'results/spray/dataset.h5'.

    Example:
        >>> imgs = np.random.rand(50, 3, 224, 224).astype(np.float32)
        >>> lbls = np.zeros(50, dtype=np.int64)
        >>> write_dataset_hdf5(imgs, lbls, 'results/spray/dataset.h5')
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('data', data=images.astype(np.float32))
        f.create_dataset('labels', data=labels.astype(np.int64))
    print(f"Dataset HDF5 written → {output_path}  ({len(images)} images)")


def write_attribution_hdf5(
    attributions: np.ndarray,
    output_path: str,
) -> None:
    """
    Write the attribution HDF5 file that ViRelAy requires.

    ViRelAy expects LRP heatmaps stored under the key 'attribution' with
    shape [N, C, H, W] (channels-first, float32).

    Args:
        attributions (np.ndarray): Heatmap array, shape [N, C, H, W], float32.
        output_path (str): Destination path, e.g. 'results/spray/attributions.h5'.

    Example:
        >>> attrs = np.random.randn(50, 3, 224, 224).astype(np.float32)
        >>> write_attribution_hdf5(attrs, 'results/spray/attributions.h5')
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('attribution', data=attributions.astype(np.float32))
    print(f"Attribution HDF5 written → {output_path}  ({len(attributions)} heatmaps)")


def write_analysis_hdf5(
    tsne_embedding: np.ndarray,
    cluster_labels: np.ndarray,
    output_path: str,
    chosen_k: int = 4,
) -> None:
    """
    Write the analysis HDF5 file that ViRelAy requires.

    This stores the CoRelAy SpRAy output: t-SNE 2-D coordinates and the
    k-means cluster assignment for each sample.

    Args:
        tsne_embedding (np.ndarray): 2-D t-SNE coords, shape [N, 2], float32.
        cluster_labels (np.ndarray): Cluster index per sample, shape [N], int.
        output_path (str): Destination path, e.g. 'results/spray/analysis.h5'.
        chosen_k (int): The k-means k value whose labels are being stored.

    Example:
        >>> tsne = np.random.randn(50, 2).astype(np.float32)
        >>> labels = np.random.randint(0, 4, size=50)
        >>> write_analysis_hdf5(tsne, labels, 'results/spray/analysis.h5', chosen_k=4)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        grp = f.create_group(f'kmeans_{chosen_k}')
        grp.create_dataset('embedding', data=tsne_embedding.astype(np.float32))
        grp.create_dataset('cluster_labels', data=cluster_labels.astype(np.int32))
    print(f"Analysis HDF5 written → {output_path}  (k={chosen_k})")


# ---------------------------------------------------------------------------
# Project YAML writer
# ---------------------------------------------------------------------------

def write_project_yaml(
    dataset_h5: str,
    attribution_h5: str,
    analysis_h5: str,
    label_map: dict,
    output_path: str = 'results/spray/project.yaml',
    chosen_k: int = 4,
) -> None:
    """
    Write the ViRelAy project YAML file.

    ViRelAy is launched with:
        virelay path/to/project.yaml

    Args:
        dataset_h5 (str): Path to dataset HDF5 file.
        attribution_h5 (str): Path to attribution HDF5 file.
        analysis_h5 (str): Path to analysis HDF5 file.
        label_map (dict): Mapping of integer label → class name string.
                          e.g. {0: 'bird', 1: 'horse'}
        output_path (str): Where to write the YAML file.
        chosen_k (int): k value used in analysis HDF5 (must match).

    Example:
        >>> write_project_yaml(
        ...     'results/spray/dataset.h5',
        ...     'results/spray/attributions.h5',
        ...     'results/spray/analysis.h5',
        ...     label_map={0: 'bird', 1: 'horse'},
        ...     output_path='results/spray/project.yaml',
        ...     chosen_k=4,
        ... )
    """
    label_lines = '\n'.join(
        f'    - label: {idx}\n      name: "{name}"'
        for idx, name in sorted(label_map.items())
    )

    yaml_content = textwrap.dedent(f"""\
        dataset:
          path: "{dataset_h5}"
          label_map:
        {label_lines}

        attributions:
          - path: "{attribution_h5}"
            key: "attribution"

        analyses:
          - path: "{analysis_h5}"
            analysis: "kmeans_{chosen_k}"
            embedding_key: "embedding"
            cluster_label_key: "cluster_labels"
    """)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    print(f"ViRelAy project YAML written → {output_path}")


# ---------------------------------------------------------------------------
# Server launcher
# ---------------------------------------------------------------------------

def launch_virelay(project_yaml: str, port: int = 8080) -> None:
    """
    Launch the ViRelAy web server.

    This starts the Flask server in a subprocess and prints the URL to open.
    Press Ctrl+C to stop it.

    Args:
        project_yaml (str): Path to the ViRelAy project YAML file.
        port (int): Port to serve on (default: 8080).

    Raises:
        FileNotFoundError: If project_yaml does not exist.
        RuntimeError: If the 'virelay' command is not found on PATH.

    Example:
        >>> launch_virelay('results/spray/project.yaml')
    """
    if not Path(project_yaml).exists():
        raise FileNotFoundError(
            f"Project YAML not found: {project_yaml}\n"
            "Run write_project_yaml() first."
        )

    print(f"\nStarting ViRelAy server on http://localhost:{port}")
    print("Open that URL in your browser, then press Ctrl+C here to stop.\n")

    try:
        subprocess.run(
            ['virelay', project_yaml, '--port', str(port)],
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "'virelay' command not found. Install with: pip install virelay"
        )


# ---------------------------------------------------------------------------
# Convenience: tensor → numpy helpers
# ---------------------------------------------------------------------------

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Detach a PyTorch tensor and convert to a NumPy float32 array.
    CoRelAy and h5py both require plain NumPy arrays.

    Args:
        tensor (torch.Tensor): Any CPU or CUDA tensor.

    Returns:
        np.ndarray: float32 NumPy array with the same shape.
    """
    return tensor.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# End-to-end convenience wrapper
# ---------------------------------------------------------------------------

def prepare_virelay_files(
    images: np.ndarray,
    labels: np.ndarray,
    attributions: np.ndarray,
    tsne_embedding: np.ndarray,
    cluster_labels: np.ndarray,
    label_map: dict,
    output_dir: str = 'results/spray',
    chosen_k: int = 4,
) -> str:
    """
    Write all three HDF5 files and the project YAML in one call.

    Returns:
        str: Path to the written project.yaml (pass this to launch_virelay).

    Example:
        >>> project_yaml = prepare_virelay_files(
        ...     images=imgs_np,
        ...     labels=labels_np,
        ...     attributions=attrs_np,
        ...     tsne_embedding=tsne_np,
        ...     cluster_labels=cluster_np,
        ...     label_map={0: 'bird', 1: 'horse'},
        ...     output_dir='results/spray',
        ...     chosen_k=4,
        ... )
        >>> launch_virelay(project_yaml)
    """
    out = Path(output_dir)

    dataset_h5    = str(out / 'dataset.h5')
    attribution_h5 = str(out / 'attributions.h5')
    analysis_h5   = str(out / 'analysis.h5')
    project_yaml  = str(out / 'project.yaml')

    write_dataset_hdf5(images, labels, dataset_h5)
    write_attribution_hdf5(attributions, attribution_h5)
    write_analysis_hdf5(tsne_embedding, cluster_labels, analysis_h5, chosen_k)
    write_project_yaml(
        dataset_h5, attribution_h5, analysis_h5,
        label_map, project_yaml, chosen_k,
    )

    print(f"\nAll ViRelAy files ready in {output_dir}/")
    print(f"To start the server, run:\n    virelay {project_yaml}")
    return project_yaml


if __name__ == '__main__':
    print("virely_method.py loaded.")
    print("ViRelAy is a Flask web server — call prepare_virelay_files()")
    print("then launch_virelay() or run: virelay path/to/project.yaml")