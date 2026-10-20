import argparse
import dataclasses
from pathlib import Path

import pytest

from lungxray.config import ConfigError, add_config_args, config_from_args, load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_default_config_loads_with_types():
    config = load_config(DEFAULT_CONFIG)
    assert config.model.name == "efficientnet_b0"
    assert isinstance(config.data.batch_size, int)
    assert isinstance(config.train.lr_head, float)
    assert isinstance(config.data.hflip, bool) and config.data.hflip is False
    assert config.to_dict()["train"]["epochs_stage2"] == config.train.epochs_stage2


def test_config_is_frozen():
    config = load_config(DEFAULT_CONFIG)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.train.seed = 1  # type: ignore[misc]


def test_overrides_are_parsed_and_coerced():
    config = load_config(
        DEFAULT_CONFIG,
        [
            "train.epochs_stage2=5",
            "train.lr_head=1e-3",
            "data.hflip=true",
            "model.name=efficientnet_b1",
        ],
    )
    assert config.train.epochs_stage2 == 5
    assert config.train.lr_head == pytest.approx(1e-3)  # "1e-3" is a string to PyYAML
    assert config.data.hflip is True
    assert config.model.name == "efficientnet_b1"


@pytest.mark.parametrize(
    "override, message",
    [
        ("train.nope=1", "Unknown config key"),
        ("nope.seed=1", "Unknown config section"),
        ("train.seed", "section.key=value"),
        ("data.batch_size=abc", "expected int"),
        ("train.imbalance=both", "train.imbalance"),
    ],
)
def test_bad_overrides_raise(override, message):
    with pytest.raises(ConfigError, match=message):
        load_config(DEFAULT_CONFIG, [override])


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(tmp_path / "missing.yaml")


def test_unknown_key_in_file_raises(tmp_path):
    text = DEFAULT_CONFIG.read_text(encoding="utf-8") + "\nextra: 1\n"
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown config key"):
        load_config(path)


def test_cli_args_round_trip():
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    args = parser.parse_args(
        ["--config", str(DEFAULT_CONFIG), "--set", "train.epochs_stage2=5", "--set", "train.seed=7"]
    )
    config = config_from_args(args)
    assert (config.train.epochs_stage2, config.train.seed) == (5, 7)
