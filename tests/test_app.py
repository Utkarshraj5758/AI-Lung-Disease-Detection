"""Streamlit app tests via AppTest, with a random-init bundle (no dataset or network)."""

import io
import logging
from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from lungxray.model import load_bundle, save_bundle

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _start(monkeypatch, weights: Path) -> AppTest:
    monkeypatch.setenv("LUNGXRAY_WEIGHTS", str(weights))
    monkeypatch.delenv("LUNGXRAY_HF_REPO", raising=False)
    st.cache_data.clear()
    st.cache_resource.clear()
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    return at


@pytest.fixture
def app(monkeypatch, bundle_path):
    return _start(monkeypatch, bundle_path)


def _upload(at: AppTest, name: str, data: bytes, mime: str) -> AppTest:
    at.file_uploader[0].upload(name, data, mime).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def _slider(at: AppTest, label: str):
    return next(s for s in at.slider if s.label == label)


def _inference_runs(caplog) -> int:
    return sum("Running inference" in r.getMessage() for r in caplog.records)


def test_landing_page(app):
    assert any("Research and education only" in w.value for w in app.warning)
    assert any("does not store uploads" in c.value for c in app.caption)
    assert "About the model" in [h.value for h in app.sidebar.header]
    assert "efficientnet_b0" in " ".join(m.value for m in app.sidebar.markdown)
    assert any("no test-set metrics" in i.value for i in app.sidebar.info)
    assert not app.metric  # nothing until an upload


def test_png_upload_and_controls_never_rerun_inference(app, sample_path, caplog):
    caplog.set_level(logging.INFO, logger="lungxray")
    _upload(app, "sample.png", sample_path.read_bytes(), "image/png")
    assert _metric(app, "Prediction") in ("NORMAL", "PNEUMONIA")
    assert _metric(app, "Decision threshold") == "0.500"
    assert _inference_runs(caplog) == 1

    _slider(app, "Heatmap opacity").set_value(0.8).run()
    app.toggle[0].set_value(True).run()
    _slider(app, "Threshold on P(pneumonia)").set_value(0.2).run()
    assert not app.exception
    assert _metric(app, "Decision threshold") == "0.200"
    assert _inference_runs(caplog) == 1


def test_dicom_upload(app, make_dicom):
    pixels = np.linspace(0, 4095, 300 * 280).reshape(300, 280).astype(np.uint16)
    _upload(app, "scan.dcm", make_dicom(pixels), "application/dicom")
    assert _metric(app, "Prediction") in ("NORMAL", "PNEUMONIA")


def test_colour_photo_gets_an_input_warning_not_a_block(app, colour_image):
    _upload(app, "photo.png", _png(colour_image), "image/png")
    assert any("strong colour" in w.value for w in app.warning)
    assert _metric(app, "Prediction") in ("NORMAL", "PNEUMONIA")


def test_unreadable_upload_shows_a_friendly_error(app):
    _upload(app, "broken.png", b"\x00" * 500, "image/png")
    errors = [e.value for e in app.error]
    assert any("could not be read" in e for e in errors)
    assert not any("Traceback" in e or "0x" in e for e in errors)


def test_missing_bundle_shows_setup_help(monkeypatch, tmp_path):
    at = _start(monkeypatch, tmp_path / "missing.pt")
    assert any("No model bundle" in e.value and "LUNGXRAY_HF_REPO" in e.value for e in at.error)


def test_sidebar_reads_metrics_from_the_bundle(monkeypatch, tmp_path, bundle_path):
    model, meta = load_bundle(bundle_path)
    meta = {
        **meta,
        "threshold": 0.9,
        "temperature": 2.0,
        "metrics": {
            "val": {"target_sensitivity": 0.95},
            "test": {
                "n": {"total": 100},
                "threshold": 0.9,
                "auroc": 0.912,
                "sensitivity": 0.8,
                "specificity": 0.7,
                "unseen_patients": {"sensitivity": 0.6, "specificity": 0.65},
            },
        },
    }
    at = _start(monkeypatch, save_bundle(tmp_path / "evaluated.pt", model, meta))
    sidebar = " ".join(m.value for m in at.sidebar.markdown)
    assert "AUROC 0.912" in sidebar and "sensitivity 80.0%" in sidebar
    assert any("not seen in training: sensitivity 60.0%" in c.value for c in at.sidebar.caption)
    assert any("95%" in m.value and "0.900" in m.value for m in at.sidebar.markdown)
