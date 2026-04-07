import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "extension/models/vgg16_chest.pth"
IMG_PATH   = "extension/data/chest_xray/pneumonia/pneumonia_000.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# SAME TRANSFORM AS TRAINING
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -------------------------------
# LOAD MODEL
# -------------------------------
def load_model():
    model = models.vgg16_bn(weights=None)
    model.classifier[6] = torch.nn.Linear(4096, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# -------------------------------
# GRAD-CAM CLASS
# -------------------------------
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer = self.model.features[-1]

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x):
        output = self.model(x)
        class_idx = torch.argmax(output)

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))

        cam -= np.min(cam)
        cam /= np.max(cam)

        return cam, class_idx.item()

# -------------------------------
# UNNORMALIZE IMAGE
# -------------------------------
def unnormalize(img_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)

    return (img * 255).astype(np.uint8)

# -------------------------------
# APPLY HEATMAP
# -------------------------------
def apply_heatmap(img, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = heatmap * 0.4 + img
    return overlay.astype(np.uint8)

# -------------------------------
# MAIN
# -------------------------------
def main():
    model = load_model()

    img = Image.open(IMG_PATH).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    gradcam = GradCAM(model)
    cam, pred = gradcam.generate(input_tensor)

    original = unnormalize(input_tensor[0])
    result = apply_heatmap(original, cam)

    label_map = {0: "Normal", 1: "Pneumonia"}

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Grad-CAM ({label_map[pred]})")
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()


if __name__ == "__main__":
    main()
