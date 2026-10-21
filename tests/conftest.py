"""Shared fixtures: a random-init bundle and synthetic images. No dataset or internet needed."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from lungxray.inference import Predictor
from lungxray.testing import make_random_bundle

ASSETS = Path(__file__).resolve().parent / "assets"
DIGITAL_XRAY_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.1.1"


@pytest.fixture(scope="session")
def bundle_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_random_bundle(tmp_path_factory.mktemp("bundle") / "random.pt")


@pytest.fixture(scope="session")
def predictor(bundle_path: Path) -> Predictor:
    return Predictor(bundle_path, device="cpu")


@pytest.fixture(scope="session")
def sample_path() -> Path:
    return ASSETS / "sample.png"


@pytest.fixture
def sample_image(sample_path: Path) -> Image.Image:
    with Image.open(sample_path) as img:
        return img.copy()


@pytest.fixture
def constant_image() -> Image.Image:
    return Image.new("L", (256, 256), 128)


@pytest.fixture
def colour_image() -> Image.Image:
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 60, size=(256, 256, 3), dtype=np.uint8)
    arr[:, :128, 0] += 180  # red left half
    arr[:, 128:, 1] += 180  # green right half
    return Image.fromarray(arr)


@pytest.fixture
def tiny_image() -> Image.Image:
    rng = np.random.default_rng(2)
    return Image.fromarray(rng.integers(0, 255, size=(64, 64), dtype=np.uint8))


# A tiny on-disk dataset in the Kermany layout, with known patient overlaps.
TINY_LAYOUT = {
    "train": {
        "NORMAL": ["IM-0001-0001.jpeg", "IM-0002-0001.jpeg"],
        "PNEUMONIA": ["person1_bacteria_1.jpeg", "person2_virus_2.jpeg"],
    },
    "val": {
        "NORMAL": ["IM-0002-0001-0002.jpeg"],  # patient IM-0002 is also in train
        "PNEUMONIA": ["person3_bacteria_3.jpeg"],
    },
    "test": {
        "NORMAL": ["IM-0003-0001.jpeg"],
        "PNEUMONIA": ["person1_virus_9.jpeg", "person3_virus_4.jpeg"],  # in train / in val
    },
}


@pytest.fixture
def tiny_dataset(tmp_path):
    for split, classes in TINY_LAYOUT.items():
        for label, names in classes.items():
            folder = tmp_path / split / label
            folder.mkdir(parents=True)
            for name in names:
                Image.new("L", (40, 30), 100).save(folder / name, format="JPEG")
    return tmp_path


@pytest.fixture
def make_dicom():
    """Factory: uncompressed greyscale DICOM bytes from (rows, cols) or (frames, rows, cols)."""

    def _make(pixels: np.ndarray, photometric: str = "MONOCHROME2", **attributes: object) -> bytes:
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.SOPClassUID = DIGITAL_XRAY_SOP_CLASS
        ds.SOPInstanceUID = generate_uid()
        bits = pixels.dtype.itemsize * 8
        ds.set_pixel_data(pixels, photometric, bits)
        for name, value in attributes.items():
            setattr(ds, name, value)
        buffer = BytesIO()
        ds.save_as(buffer, enforce_file_format=True)
        return buffer.getvalue()

    return _make
