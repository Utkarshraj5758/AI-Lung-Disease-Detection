from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from lungxray.inference import Predictor, load_image


def test_fixture_model_depends_on_input(predictor, sample_image, constant_image, colour_image):
    # Guards the random-init fixture: without BatchNorm recalibration every input gave
    # identical logits, which made the Grad-CAM and threshold tests vacuous.
    probs = {predictor.predict(img).prob_pneumonia for img in (sample_image, constant_image)}
    probs.add(predictor.predict(colour_image).prob_pneumonia)
    assert len({round(p, 4) for p in probs}) == 3, probs


def test_predict_fields(predictor, sample_image):
    pred = predictor.predict(sample_image)
    assert 0.0 <= pred.prob_pneumonia <= 1.0
    assert pred.prob_calibrated == pytest.approx(pred.prob_pneumonia)  # temperature 1.0
    assert pred.label in ("NORMAL", "PNEUMONIA")
    assert pred.threshold == 0.5


def test_threshold_logic_and_borderline(predictor, sample_image):
    p = predictor.predict(sample_image).prob_calibrated
    assert predictor.predict(sample_image, threshold=max(0.0, p - 0.2)).label == "PNEUMONIA"
    assert predictor.predict(sample_image, threshold=min(1.0, p + 0.2)).label == "NORMAL"
    assert predictor.predict(sample_image, threshold=p).label == "PNEUMONIA"  # >= threshold
    assert predictor.predict(sample_image, threshold=p).borderline
    far = 0.0 if p > 0.5 else 1.0
    assert not predictor.predict(sample_image, threshold=far).borderline
    with pytest.raises(ValueError, match="threshold"):
        predictor.predict(sample_image, threshold=1.5)


def test_temperature_and_threshold_come_from_meta(bundle_path, sample_image):
    predictor = Predictor(bundle_path, device="cpu")
    raw = predictor.predict(sample_image)
    predictor.temperature = 1000.0
    predictor.threshold = 0.25
    pred = predictor.predict(sample_image)
    assert pred.prob_pneumonia == pytest.approx(raw.prob_pneumonia)
    assert pred.prob_calibrated == pytest.approx(0.5, abs=1e-3)
    assert pred.threshold == 0.25 and pred.label == "PNEUMONIA"


def test_explain_targets(predictor, sample_image):
    by_name = predictor.explain(sample_image, target="PNEUMONIA")
    by_index = predictor.explain(sample_image, target=1)
    np.testing.assert_allclose(by_name, by_index)
    predicted = predictor.predict(sample_image).label
    np.testing.assert_allclose(
        predictor.explain(sample_image), predictor.explain(sample_image, target=predicted)
    )
    with pytest.raises(ValueError, match="Unknown target"):
        predictor.explain(sample_image, target="CAT")


def test_model_view_matches_cam(predictor, sample_image):
    view = predictor.model_view(sample_image)
    assert view.shape == (224, 224, 3) and view.dtype == np.uint8
    assert predictor.explain(sample_image).shape == view.shape[:2]


def test_input_check_is_quiet_for_greyscale_xray_like_image(predictor, sample_image):
    assert predictor.check_input(sample_image) == []


@pytest.mark.parametrize(
    "fixture, phrase",
    [
        ("constant_image", "nearly uniform"),
        ("colour_image", "colour"),
        ("tiny_image", "resolution"),
    ],
)
def test_input_check_warnings(predictor, request, fixture, phrase):
    warnings = predictor.check_input(request.getfixturevalue(fixture))
    assert any(phrase in w for w in warnings), warnings


def test_load_image_png_path_and_bytes(sample_path):
    from_path = load_image(sample_path)
    from_bytes = load_image(sample_path.read_bytes())
    assert from_path.size == from_bytes.size == (320, 288)


def test_load_image_dicom_by_magic_and_name(make_dicom):
    data = make_dicom(np.arange(64 * 80, dtype=np.uint16).reshape(64, 80))
    assert load_image(data).size == (80, 64)
    assert load_image(BytesIO(data), filename="scan.dcm").mode == "L"


def test_load_image_16_bit_png_is_scaled(tmp_path, predictor):
    arr = np.linspace(0, 65535, 300 * 300).reshape(300, 300).astype(np.uint16)
    path = tmp_path / "16bit.png"
    Image.fromarray(arr).save(path)
    img = load_image(path)
    assert img.mode.startswith("I")
    view = predictor.model_view(img)
    assert view.min() < 20 and view.max() > 235  # not clipped to white


def test_load_image_garbage_raises():
    with pytest.raises(ValueError, match="Unsupported or corrupt image"):
        load_image(b"\x00" * 300, filename="x.png")
