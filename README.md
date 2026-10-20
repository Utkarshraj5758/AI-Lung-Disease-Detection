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
pip install -e .[dev]
```

On Windows and macOS this installs CPU-only PyTorch. For a CUDA build, install
`torch` and `torchvision` from the PyTorch index first, then the package:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .[dev]
```

## Usage

```bash
lungxray predict image.png --weights artifacts/best.pt --cam overlay.png   # add --json for JSON
python scripts/check_dataset.py   # image counts per split
pytest -q                         # tests (no dataset or GPU needed)
```

`lungxray predict` needs a model bundle; `python scripts/make_dummy_bundle.py` writes a
random-init one for smoke tests. The root `train.py` and `app.py` are the legacy scripts
(they use `lung_model.pth`) and are replaced by the package in later phases.
