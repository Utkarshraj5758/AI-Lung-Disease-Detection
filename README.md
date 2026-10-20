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
python scripts/check_dataset.py     # counts, image sizes, patient overlap between splits
python -m lungxray.train            # two-stage fine-tune + calibration -> artifacts/best.pt
python -m lungxray.evaluate         # test metrics with 95% CIs and plots -> reports/
lungxray predict image.png --weights artifacts/best.pt --cam overlay.png   # add --json for JSON
pytest -q                           # tests (no dataset or GPU needed)
```

Settings live in `configs/default.yaml`; override any of them with
`--set section.key=value`. `--fast-dev-run` runs train/evaluate on synthetic data in
seconds. Install the `[train]` extra for TensorBoard logs. `python scripts/make_dummy_bundle.py`
writes a random-init bundle for smoke tests. The root `app.py` is the legacy app (it uses
`lung_model.pth`) and is replaced in a later phase.
