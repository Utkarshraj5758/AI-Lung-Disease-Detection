import torch
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("lung_model.pth", map_location=device))
model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load image
img_path = img_path = "dataset/test/NORMAL/IM-0085-0001.jpeg" # change to any image

img = Image.open(img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)

# Hook to get gradients
gradients = []
activations = []

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

def forward_hook(module, input, output):
    activations.append(output)

# Register hook on last convolution layer
target_layer = model.features[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# Forward pass
output = model(input_tensor)
pred_class = output.argmax(dim=1).item()

# Backward pass
model.zero_grad()
class_score = output[0, pred_class]
class_score.backward()

# Get pooled gradients
pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])

# Weight activations
activation = activations[0].detach()

for i in range(len(pooled_gradients)):
    activation[:, i, :, :] *= pooled_gradients[i]

heatmap = torch.mean(activation, dim=1).squeeze()
heatmap = np.maximum(heatmap.cpu(), 0)
heatmap /= torch.max(heatmap)

# Convert heatmap
heatmap = cv2.resize(heatmap.numpy(), (224, 224))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Superimpose
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
superimposed_img = heatmap * 0.4 + img

plt.imshow(cv2.cvtColor(superimposed_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.title(f"Predicted Class: {pred_class}")
plt.axis("off")
plt.show()