# Changelog

All notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [1.0.0] - 2026-10-22

First release of the rebuilt project. Results are in the README and
`reports/metrics.json`; read the limitations before using them.

### Added

- `lungxray` package (`src/lungxray`): typed YAML config with `--set` overrides, model
  bundles (weights plus preprocessing, threshold, temperature, training config, git commit,
  dataset fingerprint and metrics, loaded with `weights_only=True`), shared transforms.
- Two-stage fine-tuning of EfficientNet-B0 with weighted loss, AMP, gradient clipping,
  best-validation-AUROC checkpointing, early stopping, CSV/TensorBoard logs and
  `--fast-dev-run`.
- Temperature scaling and threshold selection on the validation split (highest threshold
  reaching the target sensitivity, plus Youden-J for reference).
- Test evaluation with seeded bootstrap 95% CIs, an unseen-patient subset, confusion
  matrix, ROC, PR, reliability and Grad-CAM example plots in `reports/`.
- Dataset audit (`scripts/check_dataset.py`): counts, image sizes, patient-id coverage and
  patient overlap between splits.
- Hook-free, thread-safe Grad-CAM; robust DICOM loading; `Predictor` with input sanity
  warnings; `lungxray predict` CLI.
- Streamlit app rebuilt on `Predictor`, with per-upload caching, a research-mode threshold
  override, a model output summary with JSON/PNG downloads, and accurate privacy and
  disclaimer text.
- FastAPI service: `GET /health`, `POST /predict` with optional Grad-CAM, upload size and
  type limits, OpenAPI docs.
- Tests (no dataset, GPU or network needed), GitHub Actions CI on Python 3.10 and 3.12,
  Dockerfile (Streamlit or API), Makefile, pre-commit config.
- Model card, rewritten README with results generated from `reports/metrics.json`
  (`scripts/render_results.py`), weight download and Hugging Face Space build scripts.

### Changed

- The class-weighted loss is now actually applied during training.
- Default decision threshold comes from the bundle (validation-tuned) instead of 0.60;
  the borderline flag uses a log-odds margin around the active threshold.
- Probabilities are labelled temperature-scaled or uncalibrated, never "calibrated"
  confidence of the chosen class.

### Fixed

- Grad-CAM overlay colours (BGR/RGB), hooks left on the shared model, overflow in the
  overlay blend, divide-by-zero on constant images.
- README merge-conflict markers; `requirements.txt` was UTF-16 with pinned local builds.
- Error messages no longer include upload file names, which reached the logs.

### Removed

- Patient age, gender and symptom inputs and the "automated radiology report" from the
  app (the model does not use them).
- Root `app.py`, `train.py` and `gradcam.py`; the plain `lung_model.pth` state dict is no
  longer used.
