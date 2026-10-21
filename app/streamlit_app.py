"""Streamlit app for the pneumonia vs normal chest X-ray classifier. Research and education only.

    streamlit run app/streamlit_app.py [-- --weights path/to/bundle.pt]

The bundle is ``--weights``, else ``$LUNGXRAY_WEIGHTS``, else ``<artifacts_dir>/best.pt``. If
that file is missing it is downloaded from the Hugging Face model repo in
``$LUNGXRAY_HF_REPO`` (or ``app.hf_repo_id`` in configs/default.yaml).

Caching: the Predictor is loaded once per process (``st.cache_resource``). Probabilities and
Grad-CAM maps are cached per upload, keyed on the SHA-256 of the file bytes and of the
bundle (``st.cache_data``), so the opacity and threshold controls never re-run the model.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image
from streamlit.errors import StreamlitAPIException

from lungxray.config import Config, load_config
from lungxray.data import to_rgb
from lungxray.gradcam import overlay
from lungxray.inference import Prediction, Predictor, load_image, output_summary
from lungxray.model import resolve_bundle

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.yaml"
UPLOAD_TYPES = ["png", "jpg", "jpeg", "dcm", "dicom"]
DISPLAY_MAX_SIDE = 1024

DISCLAIMER = (
    "**Research and education only. Not a medical device.** The model was trained on "
    "pediatric chest X-rays (children aged about 1 to 5, from one hospital in Guangzhou). It "
    "has not been validated for adults or for clinical use and must not guide anyone's care."
)
PRIVACY = (
    "This app does not store uploads; the hosting provider's logging policy applies. "
    "An uploaded image stays in memory only for your browser session; the derived "
    "probabilities and heatmaps (not the image) are cached in memory for up to an hour so "
    "the controls respond instantly."
)

log = logging.getLogger("lungxray.app")


def main() -> None:
    st.set_page_config(page_title="Chest X-ray pneumonia classifier (research)", layout="wide")
    _configure_logging()
    st.title("Chest X-ray pneumonia classifier")
    st.warning(DISCLAIMER, icon="⚠️")
    st.caption(PRIVACY)

    config = load_config(CONFIG_PATH)
    try:
        predictor = get_predictor(*_bundle_location(config))
    except FileNotFoundError as e:
        log.error("Model bundle unavailable: %s", e)
        st.error(f"**The model could not be loaded.** {e}")
        st.stop()
    except Exception:
        log.exception("Failed to load the model bundle")
        st.error(
            "**The model could not be loaded.** The bundle may be corrupt or from an "
            "incompatible version; the details are in the server log."
        )
        st.stop()

    threshold, overridden = _sidebar(predictor, config)

    upload = st.file_uploader("Chest X-ray (PNG, JPEG or DICOM)", type=UPLOAD_TYPES)
    if upload is None:
        st.info("Upload a frontal chest X-ray to see the model output and a Grad-CAM heatmap.")
        return
    data = upload.getvalue()
    file_sha256 = hashlib.sha256(data).hexdigest()
    try:
        image = load_image(data, filename=upload.name)
        result = analyze(file_sha256, predictor.bundle_sha256, _image=image, _predictor=predictor)
    except ValueError as e:
        log.warning("Rejected upload sha256=%s: %s", file_sha256[:12], e, exc_info=True)
        st.error(f"**This file could not be read.** {e}")
        return
    except Exception:
        log.exception("Analysis failed for upload sha256=%s", file_sha256[:12])
        st.error(
            "**Something went wrong while analysing this image.** The details have been "
            "logged; please try another file."
        )
        return

    prediction = predictor.prediction_from_probs(
        result["prob_raw"], result["prob_calibrated"], threshold
    )
    _results(predictor, prediction, result, image, file_sha256, overridden)


@st.cache_resource(show_spinner="Loading the model...")
def get_predictor(local_path: str, hf_repo_id: str, hf_filename: str) -> Predictor:
    bundle = resolve_bundle(local_path, hf_repo_id, hf_filename)
    predictor = Predictor(bundle)
    log.info("Loaded %s from %s on %s", predictor.model_version, bundle, predictor.device)
    return predictor


@st.cache_data(max_entries=64, ttl=3600, show_spinner="Analysing the image...")
def analyze(
    file_sha256: str, bundle_sha256: str, *, _image: Image.Image, _predictor: Predictor
) -> dict[str, Any]:
    """Threshold-independent results for one upload; the hashed arguments are the cache key."""
    log.info("Running inference for upload sha256=%s", file_sha256[:12])
    prob_raw, prob_calibrated = _predictor.probabilities(_image)
    return {
        "prob_raw": prob_raw,
        "prob_calibrated": prob_calibrated,
        # One map per class, so a threshold change that flips the label needs no new pass.
        "cams": {name: _predictor.explain(_image, target=name) for name in _predictor.class_names},
        "warnings": _predictor.check_input(_image),
    }


def _bundle_location(config: Config) -> tuple[str, str, str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--weights")
    args, _ = parser.parse_known_args(sys.argv[1:])
    local = args.weights or os.environ.get("LUNGXRAY_WEIGHTS")
    if not local:
        artifacts = Path(config.paths.artifacts_dir)
        local = (artifacts if artifacts.is_absolute() else ROOT / artifacts) / "best.pt"
    hf_repo_id = os.environ.get("LUNGXRAY_HF_REPO", config.app.hf_repo_id)
    hf_filename = os.environ.get("LUNGXRAY_HF_FILENAME", config.app.hf_filename)
    return str(local), hf_repo_id, hf_filename


def _sidebar(predictor: Predictor, config: Config) -> tuple[float, bool]:
    meta = predictor.meta
    metrics = meta.get("metrics") or {}
    training = meta.get("training_config") or {}
    pretrained = (training.get("model") or {}).get("pretrained")
    with st.sidebar:
        st.header("About the model")
        start = "ImageNet-pretrained, " if pretrained else ""
        st.markdown(
            f"- **Architecture:** {meta['model_name']} ({start}fine-tuned in two stages)\n"
            "- **Training data:** pediatric chest X-rays, NORMAL vs PNEUMONIA "
            "(Kermany et al., 2018)\n"
            f"- **Input:** {meta['image_size']}x{meta['image_size']} center crop; greyscale "
            "repeated to 3 channels\n"
            f"- **Version:** `{predictor.model_version}`, created {meta['created_at']}"
        )
        test = metrics.get("test")
        if test:
            st.markdown(
                f"**Test set** (n = {test['n']['total']}, threshold {test['threshold']:.3f}): "
                f"AUROC {_num(test.get('auroc'))}, sensitivity {_pct(test.get('sensitivity'))}, "
                f"specificity {_pct(test.get('specificity'))}."
            )
            unseen = test.get("unseen_patients")
            caveat = (
                "Many test images come from patients who also appear in the training set, so "
                "these figures are optimistic."
            )
            if unseen:
                caveat += (
                    " On test images from patients not seen in training: sensitivity "
                    f"{_pct(unseen.get('sensitivity'))}, specificity "
                    f"{_pct(unseen.get('specificity'))}."
                )
            st.caption(caveat)
        else:
            st.info(
                "This bundle has no test-set metrics. Run `python -m lungxray.evaluate` to "
                "add them."
            )
        if meta.get("note"):
            st.warning(meta["note"])
        st.markdown(f"[Model card]({config.app.model_card_url})")
        return _threshold_controls(predictor)


def _threshold_controls(predictor: Predictor) -> tuple[float, bool]:
    default = predictor.threshold
    val = (predictor.meta.get("metrics") or {}).get("val") or {}
    with st.expander("Advanced / research mode"):
        target = val.get("target_sensitivity")
        if target is not None:
            st.markdown(
                f"The default threshold **{default:.3f}** is the highest one that reached "
                f"**{target:.0%}** sensitivity on the validation set."
            )
        else:
            st.markdown(f"The default threshold is **{default:.3f}** (from the bundle).")
        st.markdown(
            "Lowering the threshold flags more images as pneumonia: fewer missed cases "
            "(higher sensitivity) but more false alarms (lower specificity). Raising it does "
            "the opposite. This control is for exploring that trade-off only."
        )
        override = st.toggle("Override the threshold", value=False)
        value = st.slider(
            "Threshold on P(pneumonia)",
            min_value=0.01,
            max_value=0.99,
            value=min(max(round(default, 3), 0.01), 0.99),
            step=0.001,
            format="%.3f",
            disabled=not override,
        )
    return (value, True) if override else (default, False)


def _results(
    predictor: Predictor,
    prediction: Prediction,
    result: dict[str, Any],
    image: Image.Image,
    file_sha256: str,
    overridden: bool,
) -> None:
    st.subheader("Model output")
    left, middle, right = st.columns(3)
    left.metric("Prediction", prediction.label)
    if predictor.temperature == 1.0:
        prob_help = "Raw model probability (no temperature has been fitted for this bundle)."
    else:
        prob_help = (
            f"Temperature-scaled with T = {predictor.temperature:.2f}, fitted on the "
            "validation set; calibration is approximate (see the model card). Raw model "
            f"probability: {_prob(prediction.prob_pneumonia)}."
        )
    middle.metric("P(pneumonia)", _prob(prediction.prob_calibrated), help=prob_help)
    right.metric(
        "Decision threshold",
        f"{prediction.threshold:.3f}",
        help="Overridden in research mode."
        if overridden
        else "Bundle default, tuned on the validation set.",
    )
    if prediction.borderline:
        low, high = predictor.borderline_band(prediction.threshold)
        st.warning(
            f"Borderline: P(pneumonia) is close to the threshold (the borderline band is "
            f"{low:.1%} to {high:.1%}), so this output is especially uncertain.",
            icon="⚠️",
        )
    for warning in result["warnings"]:
        st.warning(f"Input check: {warning}", icon="⚠️")

    st.subheader("Grad-CAM")
    opacity = st.slider("Heatmap opacity", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    blended = overlay(predictor.model_view(image), result["cams"][prediction.label], opacity)
    original, heatmap = st.columns(2)
    with original:
        _show_image(_display_copy(image), "Uploaded image")
    with heatmap:
        size = predictor.meta["image_size"]
        _show_image(blended, f"Grad-CAM for {prediction.label} on the {size}x{size} model crop")
    st.caption(
        "Red marks the regions that most increased the score of the predicted class. "
        "Grad-CAM is coarse; it is not a localization or diagnosis method."
    )

    st.subheader("Model output summary")
    summary = output_summary(
        predictor,
        prediction,
        file_sha256=file_sha256,
        input_warnings=result["warnings"],
        threshold_overridden=overridden,
    )
    st.json(summary, expanded=False)
    json_col, png_col = st.columns(2)
    json_col.download_button(
        "Download summary (JSON)",
        json.dumps(summary, indent=2),
        file_name=f"lungxray-{file_sha256[:12]}.json",
        mime="application/json",
    )
    png_col.download_button(
        "Download Grad-CAM overlay (PNG)",
        _png_bytes(blended),
        file_name=f"lungxray-{file_sha256[:12]}-gradcam.png",
        mime="image/png",
    )


def _show_image(image: Image.Image | np.ndarray, caption: str) -> None:
    try:
        st.image(image, caption=caption, width="stretch")
    except (StreamlitAPIException, TypeError, ValueError):  # Streamlit without width="stretch"
        st.image(image, caption=caption, use_container_width=True)


def _display_copy(image: Image.Image) -> Image.Image:
    rgb = to_rgb(image)
    rgb.thumbnail((DISPLAY_MAX_SIDE, DISPLAY_MAX_SIDE))
    return rgb


def _png_bytes(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def _prob(value: float) -> str:
    """Probability as a percentage that never rounds to a misleading 0% or 100%."""
    if value > 0.999:
        return ">99.9%"
    if value < 0.001:
        return "<0.1%"
    return f"{value:.1%}"


def _pct(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "n/a"


def _num(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def _configure_logging() -> None:
    package_log = logging.getLogger("lungxray")
    if not package_log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        package_log.addHandler(handler)
        package_log.setLevel(logging.INFO)


if __name__ == "__main__":
    main()
