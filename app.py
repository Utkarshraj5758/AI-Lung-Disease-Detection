import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Lung Disease Detection AI", layout="centered")

st.title("🫁 AI-Based Early Lung Disease Detection")
st.write("Upload a Chest X-ray image to detect Pneumonia.")

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load("lung_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    # =========================
    # GRAD-CAM HOOKS
    # =========================
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.features[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    # =========================
    # FORWARD PASS
    # =========================
    output = model(input_tensor)
    probs = F.softmax(output, dim=1)

    pneumonia_prob = probs[0][1].item()
    normal_prob = probs[0][0].item()

    # Adjustable threshold
    threshold = 0.60

    if pneumonia_prob > threshold:
        pred_class = 1
        confidence = pneumonia_prob
    else:
        pred_class = 0
        confidence = normal_prob

    # =========================
    # BACKWARD PASS FOR GRAD-CAM
    # =========================
    model.zero_grad()
    output[0, pred_class].backward()

    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    activation = activations[0].detach()

    for i in range(len(pooled_gradients)):
        activation[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activation, dim=1).squeeze()
    heatmap = heatmap.cpu().numpy()

    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img = np.array(image.resize((224, 224)))
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # =========================
    # DISPLAY RESULTS
    # =========================
    st.subheader("Prediction Result")

    if pred_class == 1:
        st.error(f"⚠ Pneumonia Detected")
        st.write(f"Confidence: **{confidence*100:.2f}%**")
    else:
        st.success(f"✅ Normal")
        st.write(f"Confidence: **{confidence*100:.2f}%**")

    # Borderline detection
    if 0.40 < pneumonia_prob < 0.60:
        st.warning("⚠ Borderline prediction. Image may require clinical review.")

    # Low confidence detection (possible out-of-distribution)
    if confidence < 0.55:
        st.warning("⚠ Low confidence prediction. Image may be out-of-distribution.")

    st.image(superimposed_img, caption="Grad-CAM Heatmap", use_column_width=True)

    st.warning("⚠ This AI system is for research purposes only. Not for clinical diagnosis.")