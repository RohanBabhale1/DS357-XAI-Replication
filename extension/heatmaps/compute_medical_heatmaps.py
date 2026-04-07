"""
extension/heatmaps/compute_medical_heatmaps.py

Generates LRP heatmaps for 200 chest X-ray images using:
  - EpsilonGammaBox composite (paper Listing 1, identical to Phase 2)
  - VGG-16-BN fine-tuned on chest X-ray data (from Member B)

Output: extension/heatmaps/results/
          heatmaps_medical.npy   shape (200, 3, 224, 224)
          labels_medical.npy     shape (200,)   ground truth: 0=normal 1=pneumonia
          preds_medical.npy      shape (200,)   model predictions

Usage: python extension/heatmaps/compute_medical_heatmaps.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# ── Monkey-patch: PyTorch 2.x rejects eps=0.0 that Zennit canonizer sets ──────
_orig_bn = F.batch_norm
def _patched_bn(input, running_mean, running_var, weight=None, bias=None,
                training=False, momentum=0.1, eps=1e-5):
    if eps == 0.0:
        eps = 1e-5
    return _orig_bn(input, running_mean, running_var, weight, bias,
                    training, momentum, eps)
F.batch_norm = _patched_bn

from zennit.composites import EpsilonGammaBox
from zennit.attribution import Gradient
from zennit.canonizers import SequentialMergeBatchNorm

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "extension/data/chest_xray"
MODEL_PATH = "extension/models/vgg16_chest.pth"
SAVE_DIR   = "extension/heatmaps/results"
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# EpsilonGammaBox low/high: pixel-space bounds after normalization
# Computed from (0 - mean)/std and (1 - mean)/std — same logic as paper's -3/+3
low  = torch.FloatTensor([(0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)])
high = torch.FloatTensor([(1 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)])
low  = low[None, :, None, None].expand(1, 3, 224, 224)
high = high[None, :, None, None].expand(1, 3, 224, 224)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_model():
    model = models.vgg16_bn(weights=None)
    model.classifier[6] = nn.Linear(4096, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def get_image_paths():
    paths, labels = [], []
    for label, cls in enumerate(["normal", "pneumonia"]):
        folder = os.path.join(DATA_DIR, cls)
        for fname in sorted(os.listdir(folder)):
            if fname.endswith(".png"):
                paths.append(os.path.join(folder, fname))
                labels.append(label)
    return paths, labels


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    model = load_model()
    paths, labels = get_image_paths()
    print(f"Processing {len(paths)} images with EpsilonGammaBox LRP...")

    canonizers = [SequentialMergeBatchNorm()]
    composite  = EpsilonGammaBox(low=low, high=high, canonizers=canonizers)

    heatmaps    = []
    class_preds = []

    with Gradient(model=model, composite=composite) as attributor:
        for path, label in tqdm(zip(paths, labels), total=len(paths)):
            img = Image.open(path).convert("RGB")
            x   = transform(img).unsqueeze(0)
            x.requires_grad_(True)

            out        = model(x)
            pred_class = out.argmax(dim=1).item()
            class_preds.append(pred_class)

            # One-hot target on predicted class (matches paper Listing 1)
            target = torch.zeros_like(out)
            target[0, pred_class] = 1.0

            _, relevance = attributor(x, target)
            heatmaps.append(relevance.detach().cpu().numpy()[0])  # (3, 224, 224)

    heatmaps_arr = np.array(heatmaps)   # (200, 3, 224, 224)
    labels_arr   = np.array(labels)
    preds_arr    = np.array(class_preds)

    np.save(os.path.join(SAVE_DIR, "heatmaps_medical.npy"), heatmaps_arr)
    np.save(os.path.join(SAVE_DIR, "labels_medical.npy"),   labels_arr)
    np.save(os.path.join(SAVE_DIR, "preds_medical.npy"),    preds_arr)

    accuracy = (labels_arr == preds_arr).mean()
    print(f"\n✓ heatmaps_medical.npy shape: {heatmaps_arr.shape}")
    print(f"✓ Model accuracy on 200 images: {accuracy:.3f}")
    print(f"✓ Saved to: {SAVE_DIR}/")
    


if __name__ == "__main__":
    main()