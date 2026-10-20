"""Datasets, transforms, loaders and dataset audits shared by training, evaluation and inference.

Chest X-rays are greyscale; they are converted to 3-channel RGB (the grey value
repeated) so the ImageNet-pretrained weights can be reused unchanged.

The dataset is used as laid out on disk (``<data_dir>/{train,val,test}/{NORMAL,PNEUMONIA}``);
patient overlap between the splits is measured and reported, not fixed.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import FakeData, ImageFolder

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = ("NORMAL", "PNEUMONIA")  # ImageFolder (alphabetical) order
POSITIVE_CLASS = "PNEUMONIA"
SPLITS = ("train", "val", "test")

# --fast-dev-run uses this many synthetic batches of this size per split.
FAST_DEV_BATCHES = 2
FAST_DEV_BATCH_SIZE = 4

_HIGH_BIT_DEPTH_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16N", "F"}

# Patient id = the leading prefix of the Kermany file names: NORMAL ``IM-0115-0001`` and
# ``NORMAL2-IM-0123-0001`` -> ``IM-0115`` / ``NORMAL2-IM-0123``; PNEUMONIA
# ``person1_bacteria_1`` -> ``person1``. Some files carry extra suffixes
# (``IM-0011-0001-0002``, ``person447_virus_921_1``) and belong to the same patient.
_PATIENT_PATTERNS = {
    "NORMAL": re.compile(r"^(?P<pid>(?:NORMAL2-)?IM-\d+)(?:-\d+)+$"),
    "PNEUMONIA": re.compile(r"^(?P<pid>person\d+)_(?:bacteria|virus)(?:_\d+)+$"),
}


# ---------------------------------------------------------------------------
# Pixels and transforms
# ---------------------------------------------------------------------------


def minmax_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Linearly rescale an array to the full uint8 range; constant input maps to 0."""
    arr = np.asarray(arr, dtype=np.float64)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - lo) / (hi - lo) * 255.0
    return np.round(np.nan_to_num(scaled)).astype(np.uint8)


def to_rgb(img: Image.Image) -> Image.Image:
    """Return an 8-bit RGB copy, min-max scaling 16-bit/float images first.

    PIL's own ``convert`` clips 16-bit values at 255 instead of scaling them.
    """
    if img.mode in _HIGH_BIT_DEPTH_MODES:
        img = Image.fromarray(minmax_to_uint8(np.asarray(img)))
    return img.convert("RGB")


class ToRGB:
    """Picklable transform for :func:`to_rgb` (DataLoader workers on Windows use spawn)."""

    def __call__(self, img: Image.Image) -> Image.Image:
        return to_rgb(img)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def build_train_transform(
    image_size: int,
    hflip: bool = False,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    """Mild geometric/intensity augmentation. Horizontal flip is opt-in: anatomy is asymmetric."""
    steps: list = [
        ToRGB(),
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomRotation(7),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
    ]
    if hflip:
        steps.append(transforms.RandomHorizontalFlip())
    steps += [transforms.ToTensor(), transforms.Normalize(list(mean), list(std))]
    return transforms.Compose(steps)


def build_view_transform(image_size: int, resize_size: int) -> transforms.Compose:
    """PIL -> PIL crop that the model sees, before tensor conversion and normalization.

    Grad-CAM maps line up with this crop, so overlays are drawn on it.
    """
    return transforms.Compose(
        [ToRGB(), transforms.Resize(resize_size), transforms.CenterCrop(image_size)]
    )


def build_eval_transform(
    image_size: int,
    resize_size: int,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    """Deterministic eval/inference transform: resize shorter side, center crop, normalize."""
    view = build_view_transform(image_size, resize_size)
    return transforms.Compose(
        [*view.transforms, transforms.ToTensor(), transforms.Normalize(list(mean), list(std))]
    )


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def image_folder(data_dir: str | Path, split: str, transform=None) -> ImageFolder:
    """One split of the on-disk dataset; checks the class folders."""
    root = Path(data_dir) / split
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset split not found: {root}. Expected <data_dir>/{{train,val,test}}/"
            "{NORMAL,PNEUMONIA}/ (see README)."
        )
    dataset = ImageFolder(str(root), transform=transform)
    if tuple(dataset.classes) != CLASS_NAMES:
        raise ValueError(f"{root} has classes {dataset.classes}; expected {list(CLASS_NAMES)}")
    return dataset


class FakeXrays(Dataset):
    """torchvision ``FakeData`` images with alternating labels, so every split has both classes.

    Used by ``--fast-dev-run`` so the full pipeline runs without the dataset.
    """

    classes = list(CLASS_NAMES)

    def __init__(self, size: int, image_size: int, transform=None, seed: int = 0) -> None:
        self._images = FakeData(
            size=size,
            image_size=(3, image_size + 32, image_size + 32),
            num_classes=len(CLASS_NAMES),
            random_offset=seed,
        )
        self.transform = transform
        self.targets = [i % len(CLASS_NAMES) for i in range(size)]

    def __len__(self) -> int:
        return len(self.targets)

    def raw_image(self, index: int) -> Image.Image:
        return self._images[index][0]

    def __getitem__(self, index: int):
        image = self.raw_image(index)
        return (self.transform(image) if self.transform else image), self.targets[index]


def split_dataset(
    data_dir: str | Path,
    split: str,
    transform=None,
    *,
    fake: bool = False,
    image_size: int = 224,
    seed: int = 0,
) -> Dataset:
    """The on-disk split, or a small synthetic one for ``--fast-dev-run``."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    if fake:
        size = FAST_DEV_BATCHES * FAST_DEV_BATCH_SIZE
        return FakeXrays(size, image_size, transform, seed=seed + 1000 * SPLITS.index(split))
    return image_folder(data_dir, split, transform)


def raw_image(dataset: Dataset, index: int) -> Image.Image:
    """Untransformed PIL image at ``index`` of an ImageFolder or FakeXrays dataset."""
    if isinstance(dataset, ImageFolder):
        return dataset.loader(dataset.samples[index][0])
    return dataset.raw_image(index)


def sample_patients(dataset: Dataset) -> list[str | None]:
    """Patient id per sample of an ImageFolder split (None for unparsed or synthetic data)."""
    if not isinstance(dataset, ImageFolder):
        return [None] * len(dataset)
    return [patient_id(Path(path).name, dataset.classes[t]) for path, t in dataset.samples]


# ---------------------------------------------------------------------------
# Loaders, imbalance, seeding
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and CUDA) and prefer deterministic cuDNN kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: derive NumPy/random seeds from the worker's torch seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    num_workers: int,
    device: torch.device,
    shuffle: bool = False,
    sampler: Sampler | None = None,
    seed: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def class_counts(targets: Iterable[int]) -> np.ndarray:
    return np.bincount(np.asarray(list(targets), dtype=np.int64), minlength=len(CLASS_NAMES))


def class_weights(targets: Iterable[int]) -> torch.Tensor:
    """'Balanced' CrossEntropyLoss weights ``n / (n_classes * count_c)``."""
    counts = class_counts(targets).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"Every class needs training images; got counts {counts.tolist()}")
    return torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)


def balanced_sampler(targets: Sequence[int], seed: int = 0) -> WeightedRandomSampler:
    """Sampler that draws each class equally often (with replacement), one epoch = len(targets)."""
    targets = np.asarray(targets, dtype=np.int64)
    counts = class_counts(targets)
    if (counts == 0).any():
        raise ValueError(f"Every class needs training images; got counts {counts.tolist()}")
    weights = torch.as_tensor(1.0 / counts[targets], dtype=torch.double)
    return WeightedRandomSampler(
        weights,
        num_samples=len(targets),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


# ---------------------------------------------------------------------------
# Dataset audit: patient ids, overlap, fingerprint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageRecord:
    split: str
    label: str
    path: Path
    patient: str | None


def patient_id(filename: str, label: str) -> str | None:
    """Patient id parsed from a Kermany file name, or None if it doesn't match."""
    pattern = _PATIENT_PATTERNS.get(label)
    match = pattern.match(Path(filename).stem) if pattern else None
    return match["pid"] if match else None


def scan_dataset(data_dir: str | Path) -> list[ImageRecord]:
    """Every image file under ``<data_dir>/<split>/<class>/``, with its parsed patient id."""
    root = Path(data_dir)
    records = []
    for split in SPLITS:
        for label in CLASS_NAMES:
            folder = root / split / label
            if not folder.is_dir():
                raise FileNotFoundError(
                    f"Expected folder {folder} (layout: <data_dir>/{{train,val,test}}/"
                    "{NORMAL,PNEUMONIA}/)"
                )
            for path in sorted(p for p in folder.iterdir() if p.is_file()):
                records.append(ImageRecord(split, label, path, patient_id(path.name, label)))
    return records


def patient_overlap(records: Iterable[ImageRecord]) -> dict[str, dict[str, int]]:
    """For each split pair ``a/b``: patients in both, and how many ``b`` images they account for."""
    patients: dict[str, set[str]] = {s: set() for s in SPLITS}
    images: dict[str, Counter] = {s: Counter() for s in SPLITS}
    for record in records:
        if record.patient is not None:
            patients[record.split].add(record.patient)
            images[record.split][record.patient] += 1
    overlap = {}
    for a, b in combinations(SPLITS, 2):
        shared = patients[a] & patients[b]
        overlap[f"{a}/{b}"] = {
            "shared_patients": len(shared),
            f"{a}_patients": len(patients[a]),
            f"{b}_patients": len(patients[b]),
            f"{b}_images_from_shared_patients": sum(images[b][p] for p in shared),
            f"{b}_images": sum(images[b].values()),
        }
    return overlap


def dataset_fingerprint(records: Iterable[ImageRecord]) -> str:
    """Short hash of every file's split/class/name and size (detects added/removed/moved files)."""
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda r: (r.split, r.label, r.path.name)):
        size = record.path.stat().st_size
        digest.update(f"{record.split}/{record.label}/{record.path.name}:{size}\n".encode())
    return digest.hexdigest()[:16]
