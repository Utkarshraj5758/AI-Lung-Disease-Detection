"""Typed configuration loaded from YAML, with ``--set section.key=value`` overrides.

``configs/default.yaml`` is the single source of default values: the dataclasses below
have no defaults, and unknown or missing keys are errors.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import typing
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
IMBALANCE_STRATEGIES = ("weighted_loss", "sampler")


class ConfigError(ValueError):
    """Invalid configuration file or override."""


@dataclass(frozen=True)
class PathsConfig:
    data_dir: str
    artifacts_dir: str
    runs_dir: str
    reports_dir: str


@dataclass(frozen=True)
class DataConfig:
    image_size: int
    resize_size: int
    batch_size: int
    num_workers: int
    hflip: bool


@dataclass(frozen=True)
class ModelConfig:
    name: str
    pretrained: bool


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    imbalance: str
    epochs_stage1: int
    epochs_stage2: int
    unfreeze_blocks: int
    lr_head: float
    lr_backbone: float
    weight_decay: float
    warmup_epochs: int
    grad_clip: float
    amp: bool
    patience: int


@dataclass(frozen=True)
class EvalConfig:
    target_sensitivity: float
    ece_bins: int
    bootstrap_resamples: int


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dict (e.g. for storing in a model bundle)."""
        return dataclasses.asdict(self)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH, overrides: Iterable[str] = ()) -> Config:
    """Load a YAML config and apply ``section.key=value`` overrides."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path} (run from the repo root or pass --config)"
        )
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = apply_overrides(raw, overrides)
    config = _build(Config, raw, "")
    _validate(config)
    return config


def apply_overrides(raw: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Return a copy of ``raw`` with each ``a.b=value`` applied (value parsed as YAML)."""
    raw = copy.deepcopy(raw)
    for item in overrides:
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ConfigError(f"Override must look like section.key=value, got {item!r}")
        *sections, leaf = key.split(".")
        node = raw
        for section in sections:
            if not isinstance(node.get(section), dict):
                raise ConfigError(f"Unknown config section in override {key!r}")
            node = node[section]
        if leaf not in node:
            raise ConfigError(f"Unknown config key in override {key!r}")
        node[leaf] = yaml.safe_load(value)
    return raw


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--config`` and repeatable ``--set KEY=VALUE`` options to a CLI."""
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set train.epochs_stage2=5 (repeatable)",
    )


def config_from_args(args: argparse.Namespace) -> Config:
    return load_config(args.config, args.overrides)


def _build(cls: type, raw: Any, prefix: str) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"{prefix.rstrip('.') or 'config'}: expected a mapping")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = sorted(set(raw) - set(names))
    missing = sorted(set(names) - set(raw))
    if unknown:
        raise ConfigError(f"Unknown config key(s): {', '.join(prefix + k for k in unknown)}")
    if missing:
        raise ConfigError(f"Missing config key(s): {', '.join(prefix + k for k in missing)}")
    kwargs = {}
    for name in names:
        field_type, key = hints[name], prefix + name
        if dataclasses.is_dataclass(field_type):
            kwargs[name] = _build(field_type, raw[name], key + ".")
        else:
            kwargs[name] = _coerce(raw[name], field_type, key)
    return cls(**kwargs)


def _coerce(value: Any, field_type: type, key: str) -> Any:
    if field_type is bool:
        if isinstance(value, bool):
            return value
    elif field_type is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif field_type is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):  # PyYAML reads "1e-3" (no dot) as a string
            try:
                return float(value)
            except ValueError:
                pass
    elif field_type is str:
        if isinstance(value, str):
            return value
    else:
        raise TypeError(f"Unsupported config field type {field_type!r} for {key}")
    raise ConfigError(f"{key}: expected {field_type.__name__}, got {value!r}")


def _validate(config: Config) -> None:
    if config.train.imbalance not in IMBALANCE_STRATEGIES:
        raise ConfigError(
            f"train.imbalance must be one of {IMBALANCE_STRATEGIES}, got {config.train.imbalance!r}"
        )
    if not 0.0 < config.eval.target_sensitivity <= 1.0:
        raise ConfigError("eval.target_sensitivity must be in (0, 1]")
    if config.data.resize_size < config.data.image_size:
        raise ConfigError("data.resize_size must be >= data.image_size")
