import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import pydicom
import io

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Lung Disease Detection AI", layout="wide")

st.title("🫁 AI-Based Early Lung Disease Detection")
st.info("🔒 **Privacy Notice: Zero Data Retention Policy**\n\nAll images and patient data are processed entirely in-memory and are immediately discarded after inference. No data is saved to any database or server.")
st.write("Upload a Chest X-ray image (JPG, PNG, DICOM) to detect Pneumonia.")

# =========================
# SIDEBAR: PATIENT DATA
# =========================
st.sidebar.header("Patient Information")
patient_age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
patient_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
st.sidebar.subheader("Symptoms")
symptom_cough = st.sidebar.checkbox("Cough")
symptom_fever = st.sidebar.checkbox("Fever")
symptom_fatigue = st.sidebar.checkbox("Fatigue")
symptom_diff_breathing = st.sidebar.checkbox("Difficulty Breathing")

st.sidebar.divider()
st.sidebar.header("Model Settings")
diagnostic_threshold = st.sidebar.slider("Pneumonia Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05, help="Lower threshold reduces false negatives.")

# =========================
# DEVICE & MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# HEURISTIC OOD CHECK
# =========================
def is_valid_xray(image: Image.Image) -> bool:
    """
    Basic heuristic check to see if the image is likely a chest X-ray.
    Checks if the image is largely grayscale and has sufficient variance.
    """
    img_array = np.array(image.convert("RGB"))
    
    # Calculate variance (very low variance means it's likely a blank/solid color image)
    variance = np.var(img_array)
    if variance < 100:
        return False
        
    # Check if it's grayscale-ish (R, G, B channels should be very similar)
    # Average absolute difference between channels
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    rg_diff = np.mean(np.abs(r.astype(int) - g.astype(int)))
    rb_diff = np.mean(np.abs(r.astype(int) - b.astype(int)))
    
    # If the image has high color differences, it's likely a natural picture (e.g. a cat, car)
    if rg_diff > 15 or rb_diff > 15:
         return False
         
    return True

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png", "dcm"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".dcm"):
            # Handle DICOM
            dicom_data = pydicom.dcmread(uploaded_file)
            pixel_array = dicom_data.pixel_array
            
            # Normalize to 0-255 uint8
            pixel_array = pixel_array - np.min(pixel_array)
            if np.max(pixel_array) != 0:
                pixel_array = pixel_array / np.max(pixel_array)
            pixel_array = (pixel_array * 255).astype(np.uint8)
            
            # Convert to PIL Image
            if len(pixel_array.shape) == 2:
                image = Image.fromarray(pixel_array).convert("RGB")
            else:
                image = Image.fromarray(pixel_array[0]).convert("RGB") # Take first frame if 3D
        else:
            image = Image.open(uploaded_file).convert("RGB")
            
        # Run OOD Check
        if not is_valid_xray(image):
            st.error("🚨 Error: The uploaded image does not appear to be a Chest X-ray. Please upload a valid grayscale X-ray image.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded X-ray", use_container_width=True)
            
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
            handle_forward = target_layer.register_forward_hook(forward_hook)
            handle_backward = target_layer.register_full_backward_hook(backward_hook)

            # The model is shared across sessions via st.cache_resource, so the
            # hooks must be removed even if the forward/backward pass fails.
            try:
                # =========================
                # FORWARD PASS
                # =========================
                output = model(input_tensor)

                # Raw softmax: no temperature has been fitted yet.
                probs = F.softmax(output, dim=1)
                pneumonia_prob = probs[0][1].item()

                pred_class = 1 if pneumonia_prob > diagnostic_threshold else 0

                # =========================
                # BACKWARD PASS FOR GRAD-CAM
                # =========================
                model.zero_grad()
                output[0, pred_class].backward()
            finally:
                handle_forward.remove()
                handle_backward.remove()

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
            # applyColorMap returns BGR; the PIL image it is blended with is RGB.
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # =========================
            # DISPLAY RESULTS
            # =========================
            with col2:
                st.subheader("Prediction Result")

                if pred_class == 1:
                    st.error(f"⚠ Pneumonia Detected")
                else:
                    st.success(f"✅ Normal")
                st.write(f"P(pneumonia): **{pneumonia_prob*100:.2f}%** (model probability, uncalibrated)")

                if abs(pneumonia_prob - diagnostic_threshold) < 0.10:
                    st.warning("⚠ Borderline prediction: P(pneumonia) is within 0.10 of the threshold. Image may require clinical review.")

                # Grad-CAM Opacity Slider
                st.subheader("Grad-CAM Explainability")
                opacity = st.slider("Heatmap Opacity", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
                
                img = np.array(image.resize((224, 224)))
                superimposed_img = cv2.addWeighted(img, 1.0 - opacity, heatmap, opacity, 0)

                st.image(superimposed_img, caption="Grad-CAM Heatmap", use_container_width=True)

            # =========================
            # AUTOMATED RADIOLOGY REPORT
            # =========================
            st.divider()
            st.subheader("📝 Automated Preliminary Radiology Report")
            
            symptoms_list = []
            if symptom_cough: symptoms_list.append("Cough")
            if symptom_fever: symptoms_list.append("Fever")
            if symptom_fatigue: symptoms_list.append("Fatigue")
            if symptom_diff_breathing: symptoms_list.append("Difficulty Breathing")
            
            symptoms_str = ", ".join(symptoms_list) if symptoms_list else "None reported"
            
            finding_text = "Findings concerning for Pneumonia." if pred_class == 1 else "No acute cardiopulmonary abnormalities detected. Normal lung presentation."
            
            report_markdown = f"""
            **Patient Demographics:**
            - **Age:** {patient_age}
            - **Gender:** {patient_gender}
            
            **Clinical History (Symptoms):**
            {symptoms_str}
            
            **AI Imaging Findings:**
            - The provided Chest X-ray was analyzed using EfficientNet-B0.
            - Model Output: **{'Pneumonia' if pred_class == 1 else 'Normal'}**
            - Model probability (uncalibrated), P(pneumonia): **{pneumonia_prob*100:.1f}%**
            
            **Impression:**
            {finding_text}
            
            *Note: Grad-CAM regions of interest are displayed in the heat map to guide clinical review.*
            """
            
            st.markdown(report_markdown)

            st.warning("⚠ This AI system is for research purposes only. Not for clinical diagnosis.")
            
    except Exception as e:
        st.error(f"Error processing image: {e}")