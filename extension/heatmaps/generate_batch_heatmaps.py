import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

MODEL_PATH = "extension/models/vgg16_chest.pth"
DATA_DIR = "extension/data/chest_xray"
OUT_PATH = "extension/results/heatmaps.npy"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_model():
    model = models.vgg16_bn(weights=None)
    model.classifier[6] = torch.nn.Linear(4096, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

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
        class_idx = torch.argmax(output).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        gradients = self.gradients[0].detach().cpu().numpy()
        activations = self.activations[0].detach().cpu().numpy()
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        if np.max(cam) != 0:
            cam -= np.min(cam)
            cam /= np.max(cam)
        return cam, class_idx

def main():
    model = load_model()
    gradcam = GradCAM(model)
    paths = []
    for cls in ['normal', 'pneumonia']:
        cls_dir = os.path.join(DATA_DIR, cls)
        for f in sorted(os.listdir(cls_dir)):
            if f.endswith('.png'):
                paths.append(os.path.join(cls_dir, f))
    
    heatmaps = []
    labels = []
    original_imgs = []

    print(f"Generating heatmaps for {len(paths)} images...")
    for path in tqdm(paths):
        img_pil = Image.open(path).convert("RGB")
        original_imgs.append(np.array(img_pil.resize((224, 224))))
        
        input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
        cam, pred = gradcam.generate(input_tensor)
        
        heatmaps.append(cam)
        
        # 0 for normal, 1 for pneumonia
        label = 0 if 'normal' in path else 1
        labels.append(label)
        
    heatmaps = np.array(heatmaps)
    labels = np.array(labels)
    original_imgs = np.array(original_imgs)
    
    os.makedirs("extension/results", exist_ok=True)
    np.save(OUT_PATH, heatmaps)
    np.save("extension/results/labels.npy", labels)
    np.save("extension/results/images.npy", original_imgs)
    print(f"Saved {heatmaps.shape} heatmaps to {OUT_PATH}")

if __name__ == "__main__":
    main()
