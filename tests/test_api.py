"""FastAPI service tests with a random-init bundle (no dataset or network)."""

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import MAX_UPLOAD_BYTES, create_app


@pytest.fixture(scope="module")
def client(bundle_path):
    with TestClient(create_app(bundle_path=bundle_path, device="cpu")) as test_client:
        yield test_client


def _post(client, name, data, content_type, **params):
    return client.post("/predict", files={"file": (name, data, content_type)}, params=params)


def test_health_reports_the_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["device"] == "cpu"
    assert body["model_version"].startswith("efficientnet_b0-")
    assert body["threshold"] == 0.5 and body["model"]["name"] == "efficientnet_b0"


def test_predict_png(client, sample_path):
    response = _post(client, "sample.png", sample_path.read_bytes(), "image/png")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["label"] in ("NORMAL", "PNEUMONIA")
    assert 0 <= body["prob_pneumonia"] <= 1 and 0 <= body["prob_pneumonia_raw"] <= 1
    low, high = body["borderline_band"]
    assert low < body["threshold"] < high
    assert body["gradcam_png_base64"] is None
    assert body["input_warnings"] == []
    assert len(body["file_sha256"]) == 64
    assert "not a medical device" in body["disclaimer"]


def test_predict_explain_returns_a_png_overlay(client, sample_path):
    response = _post(client, "sample.png", sample_path.read_bytes(), "image/png", explain="true")
    assert response.status_code == 200
    png = base64.b64decode(response.json()["gradcam_png_base64"])
    with Image.open(io.BytesIO(png)) as image:
        assert image.format == "PNG" and image.size == (224, 224) and image.mode == "RGB"


@pytest.mark.parametrize("content_type", ["application/dicom", "application/octet-stream"])
def test_predict_dicom(client, make_dicom, content_type):
    pixels = np.linspace(0, 4095, 256 * 240).reshape(256, 240).astype(np.uint16)
    response = _post(client, "scan.dcm", make_dicom(pixels), content_type)
    assert response.status_code == 200, response.text
    assert response.json()["label"] in ("NORMAL", "PNEUMONIA")


def test_colour_photo_is_answered_with_a_warning(client, colour_image):
    buffer = io.BytesIO()
    colour_image.save(buffer, format="JPEG")
    response = _post(client, "photo.jpg", buffer.getvalue(), "image/jpeg")
    assert response.status_code == 200
    assert any("strong colour" in w for w in response.json()["input_warnings"])


@pytest.mark.parametrize(
    "name, content_type",
    [("notes.txt", "text/plain"), ("x.png", "image/gif"), ("x.png", "application/octet-stream")],
)
def test_bad_content_type_is_415(client, sample_path, name, content_type):
    response = _post(client, name, sample_path.read_bytes(), content_type)
    assert response.status_code == 415
    assert "Unsupported content type" in response.json()["detail"]


def test_file_just_over_the_limit_is_413(client):
    response = _post(client, "big.png", b"\x00" * (MAX_UPLOAD_BYTES + 1), "image/png")
    assert response.status_code == 413
    assert "10 MB" in response.json()["detail"]


def test_body_far_over_the_limit_is_rejected_before_parsing(client):
    response = _post(client, "huge.png", b"\x00" * (MAX_UPLOAD_BYTES + 2**20), "image/png")
    assert response.status_code == 413


def test_upload_without_content_length_is_411(client):
    def chunks():
        yield b"--x\r\n"

    response = client.post(
        "/predict", content=chunks(), headers={"content-type": "multipart/form-data; boundary=x"}
    )
    assert response.status_code == 411


def test_unreadable_image_is_422_with_a_clean_message(client):
    response = _post(client, "broken.png", b"\x00" * 500, "image/png")
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "Unsupported or corrupt image" in detail and "0x" not in detail


def test_missing_file_is_422(client):
    assert client.post("/predict").status_code == 422


def test_openapi_docs(client):
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"] and "/health" in schema["paths"]
    assert "PredictResponse" in schema["components"]["schemas"]


def test_unexpected_errors_are_500_json_without_details(bundle_path, sample_path, monkeypatch):
    app = create_app(bundle_path=bundle_path, device="cpu")
    with TestClient(app, raise_server_exceptions=False) as test_client:

        def boom(*args, **kwargs):
            raise RuntimeError("secret internal detail")

        monkeypatch.setattr(app.state.predictor, "predict", boom)
        response = _post(test_client, "sample.png", sample_path.read_bytes(), "image/png")
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal error while processing the request."}


def test_error_messages_and_logs_omit_the_file_name(client, caplog):
    caplog.set_level("INFO", logger="lungxray")
    response = _post(client, "Jane_Doe_chest.png", b"\x00" * 500, "image/png")
    assert response.status_code == 422
    assert "Jane_Doe" not in response.json()["detail"]
    assert not any("Jane_Doe" in record.getMessage() for record in caplog.records)
