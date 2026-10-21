"""Command-line entry points and scripts, run the way users run them (synthetic data only)."""

import runpy
import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image

from lungxray import calibrate, evaluate, train
from lungxray.model import load_bundle

ROOT = Path(__file__).resolve().parents[1]
CONFIG = str(ROOT / "configs" / "default.yaml")


def _paths(tmp_path: Path) -> list[str]:
    args = []
    for name in ("runs_dir", "artifacts_dir", "reports_dir"):
        args += ["--set", f"paths.{name}={(tmp_path / name).as_posix()}"]
    return args


def _run_script(name: str, monkeypatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", [name, *args])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(ROOT / "scripts" / name), run_name="__main__")
    return exit_info.value.code or 0


def test_train_main_fast_dev_run(tmp_path, capsys):
    assert (
        train.main(["--fast-dev-run", "--device", "cpu", "--config", CONFIG, *_paths(tmp_path)])
        == 0
    )
    bundle = Path(capsys.readouterr().out.strip().removeprefix("Bundle: "))
    assert bundle.is_file() and "val" in load_bundle(bundle)[1]["metrics"]


def test_calibrate_main_updates_the_bundle(tmp_path, bundle_path):
    bundle = shutil.copy(bundle_path, tmp_path / "copy.pt")
    assert "val" not in load_bundle(bundle)[1]["metrics"]
    args = ["--bundle", str(bundle), "--fast-dev-run", "--device", "cpu", "--config", CONFIG]
    assert calibrate.main(args) == 0
    meta = load_bundle(bundle)[1]
    assert (
        meta["metrics"]["val"]["n"] == 8
        and meta["threshold"] == meta["metrics"]["val"]["threshold"]
    )


def test_evaluate_main_writes_reports(tmp_path, bundle_path):
    reports = tmp_path / "reports"
    args = ["--fast-dev-run", "--bundle", str(bundle_path), "--reports-dir", str(reports)]
    assert evaluate.main([*args, "--device", "cpu", "--config", CONFIG]) == 0
    assert (reports / "metrics.json").is_file() and (reports / "roc.png").is_file()


def test_check_dataset_script(tiny_dataset, monkeypatch, capsys):
    data_dir = f"paths.data_dir={tiny_dataset.as_posix()}"
    code = _run_script("check_dataset.py", monkeypatch, "--config", CONFIG, "--set", data_dir)
    out = capsys.readouterr().out
    assert code == 0
    assert "train/test 1 shared patients" in out and "overall 100.00%" in out


def test_check_dataset_script_fails_on_low_patient_id_coverage(tiny_dataset, monkeypatch, capsys):
    for i in range(3):
        Image.new("L", (40, 30)).save(tiny_dataset / "train" / "NORMAL" / f"scan{i}.jpeg")
    data_dir = f"paths.data_dir={tiny_dataset.as_posix()}"
    code = _run_script(
        "check_dataset.py", monkeypatch, "--no-sizes", "--config", CONFIG, "--set", data_dir
    )
    assert code == 1
    assert "below 95%" in capsys.readouterr().out


def test_make_dummy_bundle_script(tmp_path, monkeypatch):
    out = tmp_path / "dummy.pt"
    monkeypatch.setattr(
        sys, "argv", ["make_dummy_bundle.py", "--out", str(out), "--config", CONFIG]
    )
    runpy.run_path(str(ROOT / "scripts" / "make_dummy_bundle.py"), run_name="__main__")
    assert "random-init" in load_bundle(out)[1]["note"]
