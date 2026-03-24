"""
Generate heatmap grid for all lighthouse images x 4 attribution methods.
Produces Main Result 1 — equivalent to Figure 5 of the paper.
"""
import sys
sys.path.insert(0, '.')
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
from pathlib import Path
from xai_methods.lrp_zennit import compute_lrp, heatmap_to_image

# Setup
torch.manual_seed(42)
np.random.seed(42)

IMAGENET_SAMPLES = Path('data/imagenet_samples')
RESULTS_DIR = Path('results/heatmaps')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLASS = 437  # ImageNet lighthouse class

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load model
print("Loading VGG-16-BN...")
model = models.vgg16_bn(weights='IMAGENET1K_V1').eval()
print("Model loaded.")

# Load images
image_paths = sorted(IMAGENET_SAMPLES.glob('*.jpg'))
print(f"Found {len(image_paths)} images.")

images = []
image_names = []
for p in image_paths:
    img = Image.open(p).convert('RGB')
    images.append(transform(img).unsqueeze(0))
    image_names.append(p.stem)

# Attribution methods to run
METHODS = ['EpsilonGammaBox', 'EpsilonPlus', 'EpsilonAlpha2Beta1']

# Compute all heatmaps
print("Computing heatmaps...")
all_heatmaps = {}  # method -> list of PIL images
all_relevances = []  # for saving as .npy (needed for SpRAy)

for method in METHODS:
    print(f"  Method: {method}")
    method_heatmaps = []
    method_relevances = []
    for i, (x, name) in enumerate(zip(images, image_names)):
        relevance = compute_lrp(model, x, method, model_name='vgg16_bn', target_class=TARGET_CLASS)
        heatmap = heatmap_to_image(relevance)
        method_heatmaps.append(heatmap)
        method_relevances.append(relevance.detach().numpy())
        print(f"    {name} done.")
    all_heatmaps[method] = method_heatmaps
    all_relevances.append(np.concatenate(method_relevances, axis=0))

# Save relevances for SpRAy (use EpsilonGammaBox — paper's recommended method)
spray_relevances = all_relevances[0]  # EpsilonGammaBox
np.save('results/spray/heatmaps.npy', spray_relevances)
print(f"Saved heatmaps.npy shape: {spray_relevances.shape}")

# Build heatmap grid figure: rows = methods, columns = images
print("Building heatmap grid figure...")
n_methods = len(METHODS)
n_images = len(images)

fig, axes = plt.subplots(n_methods + 1, n_images, figsize=(n_images * 3, (n_methods + 1) * 3))

# Row 0: original images
for j, (x, name) in enumerate(zip(images, image_names)):
    # Denormalize for display
    img_np = x.squeeze(0).permute(1, 2, 0).numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)
    axes[0, j].imshow(img_np)
    axes[0, j].axis('off')
    if j == 0:
        axes[0, j].set_ylabel('Original', fontsize=9, rotation=0, labelpad=50)

# Rows 1+: heatmaps per method
for i, method in enumerate(METHODS):
    for j, heatmap in enumerate(all_heatmaps[method]):
        axes[i + 1, j].imshow(heatmap)
        axes[i + 1, j].axis('off')
    if n_images > 0:
        axes[i + 1, 0].set_ylabel(method, fontsize=9, rotation=0, labelpad=50)

plt.suptitle('LRP Heatmap Grid — VGG-16-BN (Figure 5 equivalent)', fontsize=12)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'heatmap_grid.png', dpi=150, bbox_inches='tight')
print(f"Heatmap grid saved to {RESULTS_DIR / 'heatmap_grid.png'}")