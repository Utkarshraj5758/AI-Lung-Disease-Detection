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
streamlit run app/streamlit_app.py  # web app (uses artifacts/best.pt)
uvicorn api.main:app --port 8000    # REST API (pip install -e .[api]); docs at /docs
lungxray predict image.png --weights artifacts/best.pt --cam overlay.png   # add --json for JSON
pytest -q                           # tests (no dataset or GPU needed)
```

Settings live in `configs/default.yaml`; override any of them with
`--set section.key=value`. `--fast-dev-run` runs train/evaluate on synthetic data in
seconds. Install the `[train]` extra for TensorBoard logs. `python scripts/make_dummy_bundle.py`
writes a random-init bundle for smoke tests.

The API answers `GET /health` and `POST /predict` (multipart field `file`: PNG, JPEG or
DICOM, at most 10 MB; add `?explain=true` for a base64 PNG Grad-CAM overlay), e.g.
`curl -F "file=@xray.png;type=image/png" localhost:8000/predict`.

The app takes its bundle from `-- --weights PATH`, `$LUNGXRAY_WEIGHTS` or
`artifacts/best.pt`; if none exists it downloads `best.pt` from the Hugging Face model repo
named in `$LUNGXRAY_HF_REPO` (or `app.hf_repo_id` in the config). The old `lung_model.pth`
is not a bundle and is no longer used.
