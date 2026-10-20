"""End-to-end --fast-dev-run of training, calibration and evaluation on synthetic data."""

import csv
import json
from pathlib import Path

import pytest
import torch

from lungxray.config import load_config
from lungxray.evaluate import evaluate
from lungxray.model import build_model, load_bundle
from lungxray.train import set_trainable, train, warmup_cosine

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
REPORT_FILES = (
    "metrics.json",
    "confusion_matrix.png",
    "roc.png",
    "pr.png",
    "reliability.png",
    "gradcam_examples.png",
)


@pytest.fixture
def smoke_config(tmp_path):
    overrides = [
        f"paths.{name}={(tmp_path / name).as_posix()}"
        for name in ("runs_dir", "artifacts_dir", "reports_dir")
    ]
    return load_config(DEFAULT_CONFIG, overrides)


def test_train_then_evaluate_fast_dev_run(smoke_config, tmp_path):
    bundle = train(smoke_config, device="cpu", fast_dev_run=True)
    run_dir = bundle.parent
    _, meta = load_bundle(bundle)
    assert meta["metrics"]["val"]["n"] == 8
    assert meta["metrics"]["best_epoch"]["stage"] in ("stage1", "stage2")
    assert meta["temperature"] > 0 and 0 <= meta["threshold"] <= 1
    assert meta["dataset_fingerprint"] == "fast-dev-run-fakedata"
    assert meta["training_config"]["train"]["seed"] == smoke_config.train.seed
    with (run_dir / "metrics.csv").open(encoding="utf-8") as f:
        assert [row["stage"] for row in csv.DictReader(f)] == ["stage1", "stage2"]
    assert (run_dir / "config.yaml").is_file() and (run_dir / "train.log").is_file()
    assert not (tmp_path / "artifacts_dir").exists()  # fast-dev-run never touches artifacts

    report = evaluate(smoke_config, bundle_path=bundle, device="cpu", fast_dev_run=True)
    reports = Path(report["reports_dir"])
    assert reports == run_dir / "reports"
    assert all((reports / name).is_file() for name in REPORT_FILES)
    saved = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
    assert saved["n"] == {"total": 8, "NORMAL": 4, "PNEUMONIA": 4}
    assert saved["threshold"] == meta["threshold"] and saved["fast_dev_run"] is True
    assert set(saved["confusion_matrix"]) == {"tn", "fp", "fn", "tp"}
    assert saved["validation"]["n"] == 8
    assert not (tmp_path / "reports_dir").exists()  # never the real reports folder
    assert "test" not in load_bundle(bundle)[1]["metrics"]  # caller's bundle left alone


def test_evaluate_fast_dev_run_standalone(smoke_config):
    report = evaluate(smoke_config, device="cpu", fast_dev_run=True)
    bundle = Path(report["model"]["bundle"])
    assert bundle.parent.name.endswith("fast-dev-run-eval")
    assert load_bundle(bundle)[1]["metrics"]["test"]["n"]["total"] == 8


def test_set_trainable_freezes_all_but_last_blocks():
    model = build_model(pretrained=False)
    frozen = set_trainable(model, unfreeze_blocks=3)
    assert len(frozen) == len(model.features) - 3
    for i, block in enumerate(model.features):
        assert all(p.requires_grad == (i >= len(frozen)) for p in block.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())
    assert len(set_trainable(model, 0)) == len(model.features)
    with pytest.raises(ValueError, match="unfreeze_blocks"):
        set_trainable(model, 99)


def test_warmup_cosine_schedule():
    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    scheduler = warmup_cosine(optimizer, warmup_steps=2, total_steps=10)
    lrs = []
    for _ in range(10):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert lrs[0] == pytest.approx(0.5) and lrs[1] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(lrs[1:], lrs[2:], strict=False))
    assert lrs[-1] < 0.05
