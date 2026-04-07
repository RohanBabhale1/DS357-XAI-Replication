"""
extension/models/train_chest_model.py

Fine-tunes VGG-16-BN (ImageNet pretrained) on PneumoniaMNIST subset.
Binary classification: 0=Normal, 1=Pneumonia

Output: extension/models/vgg16_chest.pth
Expected val_acc: 0.70 – 0.85 after 5 epochs
Expected runtime: ~5 min CPU, ~1 min GPU

Usage: python extension/models/train_chest_model.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

SEED = 42
DATA_DIR  = "extension/data/chest_xray"
MODEL_OUT = "extension/models/vgg16_chest.pth"
EPOCHS     = 5
BATCH_SIZE = 16
LR         = 1e-4

torch.manual_seed(SEED)
np.random.seed(SEED)


class ChestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform
        for label, cls in enumerate(["normal", "pneumonia"]):
            folder = os.path.join(root, cls)
            for fname in sorted(os.listdir(folder)):
                if fname.endswith(".png"):
                    self.samples.append((os.path.join(folder, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_model():
    model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(4096, 2)   # ImageNet 1000 → binary 2
    return model


def train():
    os.makedirs("extension/models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = ChestDataset(DATA_DIR, transform=transform)
    n_train = int(0.8 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model     = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for param in model.features.parameters():
        param.requires_grad = False
    
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += (model(imgs).argmax(dim=1) == labels).sum().item()
        val_acc = correct / len(val_ds)
        val_f1 = f1_score(labels, model(imgs).argmax(dim=1))
        val_auc = roc_auc_score(labels, model(imgs)[:, 1])
        print(f"Epoch {epoch+1}/{EPOCHS}  "
              f"loss={train_loss/len(train_loader):.4f}  "
              f"val_acc={val_acc:.3f}  "
              f"val_f1={val_f1:.3f}  "
              f"val_auc={val_auc:.3f}  ")

        if val_acc > best_val_acc :
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  → Saved best model (val_acc={val_acc:.3f})")

    print(f"\n✓ Training complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"✓ Model saved to: {MODEL_OUT}")


if __name__ == "__main__":
    train()
