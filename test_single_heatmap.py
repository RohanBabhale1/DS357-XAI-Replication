import torch
from PIL import Image
from torchvision import transforms, models
from xai_methods.lrp_zennit import compute_lrp, heatmap_to_image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load image
img = Image.open('data/imagenet_samples/lighthouse_001.jpg').convert('RGB')
x = transform(img).unsqueeze(0)

# Load model
model = models.vgg16_bn(weights='IMAGENET1K_V1').eval()

# Compute LRP with paper's recommended composite
relevance = compute_lrp(model, x, 'EpsilonGammaBox', model_name='vgg16_bn', target_class=437)
print(f"Relevance shape: {relevance.shape}")
print(f"Relevance min/max: {relevance.min():.4f} / {relevance.max():.4f}")

# Save heatmap
heatmap = heatmap_to_image(relevance)
heatmap.save('results/heatmaps/test_single.png')
print("Heatmap saved to results/heatmaps/test_single.png")