"""
extension/data/download_chest_data.py

Downloads PneumoniaMNIST from MedMNIST benchmark.
Reference: Yang et al. 2023, MedMNIST v2 (Scientific Data)
Original images: Kermany et al. 2018 (Cell) — pediatric chest X-rays
License: CC BY 4.0

Output: extension/data/chest_xray/
          normal/normal_000.png ... normal_099.png
          pneumonia/pneumonia_000.png ... pneumonia_099.png

Usage: python extension/data/download_chest_data.py
"""

import os
import numpy as np
from PIL import Image
from medmnist import PneumoniaMNIST

SAVE_DIR = "data/chest_xray"
N_PER_CLASS = 100
SEED = 42


def main():
    os.makedirs(f"{SAVE_DIR}/normal", exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/pneumonia", exist_ok=True)

    print("Loading PneumoniaMNIST (train split, 224x224)...")

    # IMPORTANT: download=False since file is manually placed
    dataset = PneumoniaMNIST(split='train', download=True, size=28)

    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(dataset))

    normal_saved = 0
    pneumonia_saved = 0

    for idx in indices:
        img, label = dataset[int(idx)]
        label = int(label[0])

        # Convert to RGB (required for CNN models like VGG16)
        img_rgb = Image.fromarray(np.array(img)).convert("RGB")

        if label == 0 and normal_saved < N_PER_CLASS:
            img_rgb.save(f"{SAVE_DIR}/normal/normal_{normal_saved:03d}.png")
            normal_saved += 1

        elif label == 1 and pneumonia_saved < N_PER_CLASS:
            img_rgb.save(f"{SAVE_DIR}/pneumonia/pneumonia_{pneumonia_saved:03d}.png")
            pneumonia_saved += 1

        if normal_saved >= N_PER_CLASS and pneumonia_saved >= N_PER_CLASS:
            break

    print(f"✓ Saved {normal_saved} normal images")
    print(f"✓ Saved {pneumonia_saved} pneumonia images")
    print(f"✓ Total: {normal_saved + pneumonia_saved} images in {SAVE_DIR}/")


if __name__ == "__main__":
    main()