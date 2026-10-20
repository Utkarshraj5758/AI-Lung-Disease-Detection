"""Two-stage fine-tuning.

    python -m lungxray.train [--set train.epochs_stage2=5] [--fast-dev-run]

Stage 1 trains the head on a frozen backbone (backbone BatchNorm stays in eval mode).
Stage 2 unfreezes the last ``train.unfreeze_blocks`` feature blocks with a lower learning
rate and a warmup + cosine schedule. Each epoch is scored on the validation split; the best
validation AUROC is saved as ``runs/<timestamp>/best.pt`` (early stopping after
``train.patience`` epochs without improvement), then calibrated on validation (temperature +
threshold, see calibrate.py) and copied to ``<artifacts_dir>/best.pt``.

``--fast-dev-run`` runs the whole loop on 2 synthetic FakeData batches per stage with
random-init weights: no dataset, GPU or download needed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import math
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from lungxray.calibrate import calibrate_bundle, softmax_np
from lungxray.config import Config, add_config_args, config_from_args
from lungxray.data import (
    CLASS_NAMES,
    FAST_DEV_BATCH_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    POSITIVE_CLASS,
    balanced_sampler,
    build_eval_transform,
    build_train_transform,
    class_counts,
    class_weights,
    dataset_fingerprint,
    make_loader,
    patient_overlap,
    scan_dataset,
    seed_everything,
    split_dataset,
)
from lungxray.metrics import binary_metrics
from lungxray.model import build_meta, build_model, collect_logits, resolve_device, save_bundle

log = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Two-stage fine-tuning of the classifier.")
    parser.add_argument("--device", help="torch device (default: cuda if available)")
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="2 synthetic batches per stage, random-init weights (no dataset or download)",
    )
    add_config_args(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)
    bundle = train(config, device=args.device, fast_dev_run=args.fast_dev_run)
    print(f"Bundle: {bundle}")
    return 0


def train(
    config: Config, *, device: str | torch.device | None = None, fast_dev_run: bool = False
) -> Path:
    """Run both stages, calibrate the best epoch, and return the final bundle path."""
    seed_everything(config.train.seed)
    device = resolve_device(device)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config.paths.runs_dir) / (stamp + ("-fast-dev-run" if fast_dev_run else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    handler = _log_to_file(run_dir / "train.log")
    try:
        return _train(config, device, run_dir, fast_dev_run)
    finally:
        logging.getLogger("lungxray").removeHandler(handler)
        handler.close()


def _train(config: Config, device: torch.device, run_dir: Path, fast_dev_run: bool) -> Path:
    cfg_data, cfg_train = config.data, config.train
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    log.info("Run directory: %s | device: %s | fast_dev_run: %s", run_dir, device, fast_dev_run)

    batch_size = FAST_DEV_BATCH_SIZE if fast_dev_run else cfg_data.batch_size
    num_workers = 0 if fast_dev_run else cfg_data.num_workers
    train_ds = split_dataset(
        config.paths.data_dir,
        "train",
        build_train_transform(cfg_data.image_size, cfg_data.hflip),
        fake=fast_dev_run,
        image_size=cfg_data.image_size,
        seed=cfg_train.seed,
    )
    val_ds = split_dataset(
        config.paths.data_dir,
        "val",
        build_eval_transform(cfg_data.image_size, cfg_data.resize_size),
        fake=fast_dev_run,
        image_size=cfg_data.image_size,
        seed=cfg_train.seed,
    )
    if fast_dev_run:
        fingerprint = "fast-dev-run-fakedata"
    else:
        records = scan_dataset(config.paths.data_dir)
        fingerprint = dataset_fingerprint(records)
        for pair, stats in patient_overlap(records).items():
            log.info("Patient overlap %s: %s", pair, stats)
    targets = list(train_ds.targets)
    log.info("Train class counts %s: %s", list(CLASS_NAMES), class_counts(targets).tolist())

    # Class imbalance: weighted loss or balanced sampling, never both.
    if cfg_train.imbalance == "sampler":
        sampler = balanced_sampler(targets, seed=cfg_train.seed)
        criterion = nn.CrossEntropyLoss()
    else:
        sampler = None
        criterion = nn.CrossEntropyLoss(weight=class_weights(targets).to(device))
    log.info("Imbalance strategy: %s", cfg_train.imbalance)
    train_loader = make_loader(
        train_ds,
        batch_size,
        num_workers=num_workers,
        device=device,
        shuffle=True,
        sampler=sampler,
        seed=cfg_train.seed,
    )
    val_loader = make_loader(val_ds, batch_size, num_workers=num_workers, device=device)

    pretrained = config.model.pretrained and not fast_dev_run
    model = build_model(config.model.name, len(CLASS_NAMES), pretrained=pretrained).to(device)
    meta_base = {
        "model_name": config.model.name,
        "class_names": list(CLASS_NAMES),
        "positive_class": POSITIVE_CLASS,
        "image_size": cfg_data.image_size,
        "resize_size": cfg_data.resize_size,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "training_config": config.to_dict(),
        "dataset_fingerprint": fingerprint,
    }
    use_amp = cfg_train.amp and device.type == "cuda"
    state = _RunState(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        use_amp=use_amp,
        scaler=_grad_scaler(use_amp),
        grad_clip=cfg_train.grad_clip,
        patience=cfg_train.patience,
        best_path=run_dir / "best.pt",
        meta_base=meta_base,
        logger=_MetricsLogger(run_dir),
    )
    epochs = (1, 1) if fast_dev_run else (cfg_train.epochs_stage1, cfg_train.epochs_stage2)
    try:
        # Stage 1: head only.
        frozen = set_trainable(model, unfreeze_blocks=0)
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(), lr=cfg_train.lr_head, weight_decay=cfg_train.weight_decay
        )
        _run_stage(state, "stage1", epochs[0], frozen, optimizer, scheduler=None)

        # Stage 2: last N blocks + head, discriminative LRs, warmup + cosine.
        frozen = set_trainable(model, unfreeze_blocks=cfg_train.unfreeze_blocks)
        backbone = [p for p in model.features.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone, "lr": cfg_train.lr_backbone},
                {"params": list(model.classifier.parameters()), "lr": cfg_train.lr_head},
            ],
            weight_decay=cfg_train.weight_decay,
        )
        steps = len(train_loader)
        scheduler = warmup_cosine(optimizer, cfg_train.warmup_epochs * steps, epochs[1] * steps)
        _run_stage(state, "stage2", epochs[1], frozen, optimizer, scheduler)
    finally:
        state.logger.close()

    log.info("Best val AUROC %.4f; calibrating %s", state.best_score, state.best_path)
    calibrate_bundle(state.best_path, config, device=device, fast_dev_run=fast_dev_run)
    if fast_dev_run:
        return state.best_path
    final = Path(config.paths.artifacts_dir) / "best.pt"
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(state.best_path, final)
    log.info("Copied best bundle to %s", final)
    return final


def set_trainable(model: nn.Module, unfreeze_blocks: int) -> list[nn.Module]:
    """Freeze all but the last ``unfreeze_blocks`` feature blocks; the head stays trainable.

    Returns the frozen blocks, which are kept in eval mode during training so their
    BatchNorm statistics don't drift.
    """
    blocks = list(model.features)
    if not 0 <= unfreeze_blocks <= len(blocks):
        raise ValueError(f"unfreeze_blocks must be in [0, {len(blocks)}], got {unfreeze_blocks}")
    n_frozen = len(blocks) - unfreeze_blocks
    for i, block in enumerate(blocks):
        for p in block.parameters():
            p.requires_grad = i >= n_frozen
    for p in model.classifier.parameters():
        p.requires_grad = True
    return blocks[:n_frozen]


def warmup_cosine(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Per-step LR factor: linear warmup to 1, then cosine decay to 0 at ``total_steps``."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, factor)


class _RunState:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)
        self.best_score = -math.inf
        self.saved = False


def _run_stage(
    state: _RunState,
    name: str,
    epochs: int,
    frozen: list[nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR | None,
) -> None:
    stale = 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss = _train_one_epoch(state, frozen, optimizer, scheduler, f"{name} {epoch}")
        val = _validate(state)
        lrs = [group["lr"] for group in optimizer.param_groups]
        state.logger.log(
            {
                "stage": name,
                "epoch": epoch,
                "train_loss": train_loss,
                **val,
                "lr_head": lrs[-1],
                "lr_backbone": lrs[0] if len(lrs) > 1 else 0.0,
                "seconds": time.perf_counter() - started,
            }
        )
        score = val["val_auroc"] if np.isfinite(val["val_auroc"]) else -math.inf
        if not state.saved or score > state.best_score:
            state.best_score, state.saved, stale = score, True, 0
            best_epoch = {"stage": name, "epoch": epoch, **val}
            meta = build_meta(**state.meta_base, metrics={"best_epoch": best_epoch})
            save_bundle(state.best_path, state.model, meta)
            log.info("New best val AUROC %.4f (%s epoch %d) saved", score, name, epoch)
        else:
            stale += 1
            if stale >= state.patience:
                log.info("Early stopping %s after %d epochs without improvement", name, stale)
                break


def _train_one_epoch(
    state: _RunState,
    frozen: list[nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR | None,
    desc: str,
) -> float:
    model, device = state.model, state.device
    model.train()
    for block in frozen:
        block.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    total, count = 0.0, 0
    for x, y in tqdm(state.train_loader, desc=desc, leave=False, disable=None):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=state.use_amp):
            loss = state.criterion(model(x), y)
        state.scaler.scale(loss).backward()
        state.scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(params, state.grad_clip)
        scale = state.scaler.get_scale()
        state.scaler.step(optimizer)
        state.scaler.update()
        # AMP skips the optimizer step on inf/NaN gradients (and lowers the scale); keep the
        # LR schedule in step with the updates that actually happened.
        skipped = state.use_amp and state.scaler.get_scale() < scale
        if scheduler is not None and not skipped:
            scheduler.step()
        total += loss.item() * x.size(0)
        count += x.size(0)
    return total / max(count, 1)


def _validate(state: _RunState) -> dict[str, float]:
    logits, labels = collect_logits(state.model, state.val_loader, state.device)
    loss = state.criterion(
        torch.from_numpy(logits).to(state.device), torch.from_numpy(labels).to(state.device)
    )
    positive = CLASS_NAMES.index(POSITIVE_CLASS)
    probs = softmax_np(logits)[:, positive]
    m = binary_metrics((labels == positive).astype(int), probs, threshold=0.5)
    return {
        "val_loss": float(loss),
        "val_auroc": m["auroc"],
        "val_sensitivity": m["sensitivity"],
        "val_specificity": m["specificity"],
    }


def _grad_scaler(enabled: bool):
    if hasattr(torch.amp, "GradScaler"):  # torch >= 2.3
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


class _MetricsLogger:
    """One row per epoch to ``metrics.csv`` and, if installed, TensorBoard."""

    def __init__(self, run_dir: Path) -> None:
        self._file = (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8")
        self._csv: csv.DictWriter | None = None
        self._step = 0
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            log.warning("tensorboard is not installed; logging CSV only (pip install -e .[train])")
            self._tb = None
        else:
            self._tb = SummaryWriter(str(run_dir / "tensorboard"))

    def log(self, row: dict[str, Any]) -> None:
        if self._csv is None:
            self._csv = csv.DictWriter(self._file, fieldnames=list(row))
            self._csv.writeheader()
        self._csv.writerow(row)
        self._file.flush()
        if self._tb is not None:
            for key, value in row.items():
                if isinstance(value, (int, float)) and key != "epoch":
                    self._tb.add_scalar(key, value, self._step)
        self._step += 1
        log.info(
            "%s epoch %d | train_loss %.4f | val_loss %.4f | val_auroc %.4f | sens %.3f | "
            "spec %.3f | %.0fs",
            row["stage"],
            row["epoch"],
            row["train_loss"],
            row["val_loss"],
            row["val_auroc"],
            row["val_sensitivity"],
            row["val_specificity"],
            row["seconds"],
        )

    def close(self) -> None:
        self._file.close()
        if self._tb is not None:
            self._tb.close()


def _log_to_file(path: Path) -> logging.Handler:
    package_log = logging.getLogger("lungxray")
    package_log.setLevel(logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    package_log.addHandler(handler)
    return handler


if __name__ == "__main__":
    raise SystemExit(main())
