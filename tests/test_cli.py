import json

from PIL import Image

from lungxray.cli import main


def test_predict_json_with_cam(bundle_path, sample_path, tmp_path, capsys):
    cam_path = tmp_path / "out" / "cam.png"
    code = main(
        [
            "predict",
            str(sample_path),
            "--weights",
            str(bundle_path),
            "--json",
            "--cam",
            str(cam_path),
        ]
    )
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["prediction"]["label"] in ("NORMAL", "PNEUMONIA")
    assert result["prediction"]["threshold"] == 0.5
    assert result["input_warnings"] == []
    assert result["cam"] == str(cam_path)
    assert result["model"]["model_name"] == "efficientnet_b0"
    with Image.open(cam_path) as cam:
        assert cam.size == (224, 224) and cam.mode == "RGB"


def test_predict_text_output(bundle_path, sample_path, capsys):
    assert main(["predict", str(sample_path), "--weights", str(bundle_path)]) == 0
    out = capsys.readouterr().out
    assert "P(pneumonia)=" in out and "uncalibrated" in out


def test_missing_weights_is_a_clean_error(sample_path, tmp_path, capsys):
    code = main(["predict", str(sample_path), "--weights", str(tmp_path / "missing.pt")])
    assert code == 1
    assert "Model bundle not found" in capsys.readouterr().err
