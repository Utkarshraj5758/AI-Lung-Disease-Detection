"""Docs tooling: generated results stay in sync with reports/metrics.json; release scripts work."""

import json
import runpy
from pathlib import Path

import pytest

from lungxray.model import load_bundle

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ["--config", str(ROOT / "configs" / "default.yaml")]
render = runpy.run_path(str(ROOT / "scripts" / "render_results.py"))
space = runpy.run_path(str(ROOT / "scripts" / "build_space.py"))
download = runpy.run_path(str(ROOT / "scripts" / "download_weights.py"))


def test_readme_and_model_card_match_metrics_json():
    # Fails when a number in the docs no longer comes from reports/metrics.json.
    # Fix with `python scripts/render_results.py` (or `make docs`).
    assert render["main"](["--check"]) == 0


def test_every_generated_number_comes_from_the_report():
    report = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    auroc = report["metrics"]["auroc"]
    assert render["fmt"](auroc, False) in readme
    assert f"{report['metrics']['sensitivity']['value']:.1%}" in readme


def test_formatting():
    fmt = render["fmt"]
    assert fmt({"value": 0.98765, "ci95": [0.97, 0.99]}, False) == "0.988 (0.970–0.990)"
    assert fmt({"value": 0.914, "ci95": [0.888, 0.94]}, True) == "91.4% (88.8–94.0)"
    assert fmt({"value": 0.5, "ci95": None}, False) == "0.500"
    assert fmt({"value": None, "ci95": None}, True) == "n/a"


def test_missing_metrics_render_as_tbd(tmp_path):
    assert render["load_report"](tmp_path / "missing.json") is None
    assert render["readme_results"](None) == render["model_card_results"](None) == render["TBD"]


def test_fast_dev_run_metrics_are_refused(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"fast_dev_run": True}), encoding="utf-8")
    with pytest.raises(SystemExit, match="fast-dev-run"):
        render["load_report"](path)


def test_replace_block():
    text = "a\n<!-- BEGIN GENERATED: x -->\nold\n<!-- END GENERATED: x -->\nb\n"
    replaced = render["replace_block"](text, "x", "new")
    assert replaced == "a\n<!-- BEGIN GENERATED: x -->\nnew\n<!-- END GENERATED: x -->\nb\n"
    with pytest.raises(SystemExit, match="markers"):
        render["replace_block"]("no markers", "x", "new")


def test_build_space(tmp_path):
    out = tmp_path / "space"
    space["build_space"](out, "user/model", "1.66.0")
    for name in ("app/streamlit_app.py", "src/lungxray/inference.py", "pyproject.toml", "LICENSE"):
        assert (out / name).is_file(), name
    assert not list(out.rglob("__pycache__"))
    assert 'hf_repo_id: "user/model"' in (out / "configs" / "default.yaml").read_text("utf-8")
    requirements = (out / "requirements.txt").read_text("utf-8").splitlines()
    assert requirements[0] == "--extra-index-url https://download.pytorch.org/whl/cpu"
    assert requirements[-1] == "." and "torch>=2.2" in requirements
    readme = (out / "README.md").read_text("utf-8")
    assert readme.startswith("---\n") and "sdk: streamlit" in readme
    assert "app_file: app/streamlit_app.py" in readme and "sdk_version: 1.66.0" in readme


@pytest.fixture
def fake_hub(monkeypatch, bundle_path):
    import huggingface_hub

    calls = []

    def fake_download(repo_id, filename, revision=None, **kwargs):
        calls.append((repo_id, filename, revision))
        return str(bundle_path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.delenv("LUNGXRAY_HF_REPO", raising=False)
    return calls


def test_download_weights(tmp_path, fake_hub, capsys):
    out = tmp_path / "artifacts" / "best.pt"
    assert download["main"](["--repo-id", "user/model", "--out", str(out), *CONFIG]) == 0
    assert fake_hub == [("user/model", "best.pt", None)]
    assert load_bundle(out)[1]["model_name"] == "efficientnet_b0"
    assert "Saved efficientnet_b0-" in capsys.readouterr().out
    # An existing file is kept unless --force is given.
    assert download["main"](["--repo-id", "user/model", "--out", str(out), *CONFIG]) == 1
    assert download["main"](["--repo-id", "user/model", "--out", str(out), "--force", *CONFIG]) == 0


def test_download_weights_rejects_a_file_that_is_not_a_bundle(tmp_path, monkeypatch):
    import huggingface_hub

    junk = tmp_path / "junk.pt"
    junk.write_bytes(b"not a bundle")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(junk))
    out = tmp_path / "best.pt"
    assert download["main"](["--repo-id", "user/model", "--out", str(out), *CONFIG]) == 1
    assert not out.exists()


def test_download_weights_needs_a_repo(tmp_path, monkeypatch):
    monkeypatch.delenv("LUNGXRAY_HF_REPO", raising=False)
    with pytest.raises(SystemExit):
        download["main"](["--out", str(tmp_path / "best.pt"), *CONFIG])
