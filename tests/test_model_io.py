from pathlib import Path

import numpy as np
import pytest
import torch

from lungxray.model import (
    REQUIRED_META_KEYS,
    build_meta,
    build_model,
    bundle_source,
    load_bundle,
    resolve_bundle,
    save_bundle,
)


def _meta(**overrides):
    meta = build_meta(
        model_name="efficientnet_b0",
        class_names=["NORMAL", "PNEUMONIA"],
        image_size=224,
        resize_size=256,
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
        threshold=0.42,
        temperature=1.7,
        training_config={"train": {"seed": 1}},
        dataset_fingerprint="abc123",
        metrics={"auroc": None},
    )
    meta.update(overrides)
    return meta


def test_round_trip_preserves_weights_and_meta(tmp_path):
    torch.manual_seed(0)
    model = build_model(pretrained=False)
    meta = _meta()
    path = save_bundle(tmp_path / "sub" / "bundle.pt", model, meta)

    loaded, loaded_meta = load_bundle(path)
    assert loaded_meta == meta
    assert not loaded.training
    original = model.state_dict()
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, original[key]), key
    assert not list(tmp_path.glob("**/*.tmp"))


def test_meta_has_required_keys():
    meta = _meta()
    assert set(REQUIRED_META_KEYS) <= meta.keys()
    assert meta["created_at"].endswith("+00:00")


def test_missing_file_fails_loudly(tmp_path):
    missing = tmp_path / "nope.pt"
    with pytest.raises(FileNotFoundError, match="Model bundle not found") as info:
        load_bundle(missing)
    assert str(missing) in str(info.value)


def test_plain_state_dict_is_rejected(tmp_path):
    path = tmp_path / "legacy.pth"
    torch.save(build_model(pretrained=False).state_dict(), path)
    with pytest.raises(ValueError, match="not a lungxray bundle"):
        load_bundle(path)


class _Custom:
    pass


def test_loading_uses_weights_only(tmp_path):
    path = tmp_path / "unsafe.pt"
    torch.save({"format_version": 1, "state_dict": {}, "meta": {"x": _Custom()}}, path)
    with pytest.raises(ValueError, match="Could not load model bundle"):
        load_bundle(path)


@pytest.mark.parametrize("value", [np.float64(0.5), {1: "int key"}, object()])
def test_non_plain_meta_is_rejected(tmp_path, value):
    with pytest.raises(TypeError, match="meta"):
        save_bundle(tmp_path / "b.pt", build_model(pretrained=False), _meta(threshold=value))


def test_missing_meta_key_is_rejected(tmp_path):
    meta = _meta()
    del meta["temperature"]
    with pytest.raises(ValueError, match="temperature"):
        save_bundle(tmp_path / "b.pt", build_model(pretrained=False), meta)


def test_build_model_head_and_unsupported_names():
    model = build_model("efficientnet_b0", num_classes=3, pretrained=False)
    assert model.classifier[-1].out_features == 3
    with pytest.raises(ValueError, match="Unsupported model"):
        build_model("resnet50", pretrained=False)
    with pytest.raises(ValueError, match="Unknown torchvision model"):
        build_model("efficientnet_zz", pretrained=False)


def test_positive_class_must_be_a_class():
    with pytest.raises(ValueError, match="positive_class"):
        build_meta(
            model_name="efficientnet_b0",
            class_names=["A", "B"],
            image_size=224,
            resize_size=256,
            mean=(0, 0, 0),
            std=(1, 1, 1),
        )


def test_resolve_bundle_prefers_the_local_file(bundle_path):
    assert resolve_bundle(bundle_path, "user/repo") == bundle_path


def test_resolve_bundle_without_local_file_or_repo_explains_setup(tmp_path):
    with pytest.raises(FileNotFoundError, match="LUNGXRAY_HF_REPO"):
        resolve_bundle(tmp_path / "missing.pt")


def test_resolve_bundle_downloads_from_the_hub(monkeypatch, tmp_path, bundle_path):
    import huggingface_hub

    calls = []

    def fake_download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename))
        return str(bundle_path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    assert resolve_bundle(tmp_path / "missing.pt", "user/repo", "best.pt") == bundle_path
    assert calls == [("user/repo", "best.pt")]


def test_resolve_bundle_reports_download_failures(monkeypatch, tmp_path):
    import huggingface_hub

    def offline(**kwargs):
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    with pytest.raises(FileNotFoundError, match="user/repo"):
        resolve_bundle(tmp_path / "missing.pt", "user/repo")


def test_bundle_source_precedence(monkeypatch, tmp_path):
    from lungxray.config import load_config

    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    for name in ("LUNGXRAY_WEIGHTS", "LUNGXRAY_HF_REPO", "LUNGXRAY_HF_FILENAME"):
        monkeypatch.delenv(name, raising=False)
    local, repo, filename = bundle_source(config, root=tmp_path)
    assert Path(local) == tmp_path / config.paths.artifacts_dir / "best.pt"
    assert (repo, filename) == (config.app.hf_repo_id, config.app.hf_filename)

    monkeypatch.setenv("LUNGXRAY_WEIGHTS", "env.pt")
    monkeypatch.setenv("LUNGXRAY_HF_REPO", "user/repo")
    monkeypatch.setenv("LUNGXRAY_HF_FILENAME", "model.pt")
    assert bundle_source(config, root=tmp_path) == ("env.pt", "user/repo", "model.pt")
    assert bundle_source(config, weights="cli.pt")[0] == "cli.pt"  # explicit beats env
