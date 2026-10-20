# AI-Lung-Disease-Detection

Deep learning system for detecting pneumonia in chest X-ray images, using
EfficientNet-B0 transfer learning and Grad-CAM explanations.

> **Research and education only.** Not a medical device and not validated for clinical use.

## Model performance

| Metric | Value |
| --- | --- |
| Test accuracy | TBD (run `make evaluate`) |
| Pneumonia recall | TBD (run `make evaluate`) |
| False negatives | TBD (run `make evaluate`) |

## Features

- Transfer learning (EfficientNet-B0)
- Confusion matrix and evaluation metrics
- Grad-CAM visualization
- Streamlit web application

## Installation

```bash
pip install -r requirements.txt
```

On Windows and macOS this installs CPU-only PyTorch. For a CUDA build, install
`torch` and `torchvision` from the PyTorch index first, then the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Usage

```bash
python train.py              # train and save lung_model.pth
streamlit run app.py         # launch the web app
python gradcam.py            # Grad-CAM for a single image
```
