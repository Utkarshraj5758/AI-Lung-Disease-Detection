import pickle
from collections import Counter

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from lungxray.data import (
    ToRGB,
    balanced_sampler,
    build_eval_transform,
    build_train_transform,
    class_weights,
    dataset_fingerprint,
    image_folder,
    make_loader,
    minmax_to_uint8,
    patient_id,
    patient_overlap,
    raw_image,
    sample_patients,
    scan_dataset,
    seed_everything,
    split_dataset,
    to_rgb,
)


@pytest.mark.parametrize("mode", ["L", "RGB", "I;16"])
def test_eval_transform_outputs_3x224x224(mode):
    rng = np.random.default_rng(0)
    if mode == "I;16":
        img = Image.fromarray(rng.integers(0, 4096, size=(300, 400), dtype=np.uint16))
    elif mode == "RGB":
        img = Image.fromarray(rng.integers(0, 255, size=(300, 400, 3), dtype=np.uint8))
    else:
        img = Image.fromarray(rng.integers(0, 255, size=(300, 400), dtype=np.uint8))
    x = build_eval_transform(224, 256)(img)
    assert tuple(x.shape) == (3, 224, 224)


def test_train_transform_shape_and_flip_is_opt_in(sample_image):
    default = build_train_transform(224)
    assert tuple(default(sample_image).shape) == (3, 224, 224)
    assert not any(isinstance(t, transforms.RandomHorizontalFlip) for t in default.transforms)
    flipped = build_train_transform(224, hflip=True)
    assert any(isinstance(t, transforms.RandomHorizontalFlip) for t in flipped.transforms)


def test_to_rgb_scales_16_bit():
    img = Image.fromarray(np.array([[0, 30000], [45000, 65535]], dtype=np.uint16))
    rgb = np.asarray(to_rgb(img))
    assert rgb.shape == (2, 2, 3)
    assert rgb[0, 0, 0] == 0 and rgb[1, 1, 0] == 255
    assert (rgb[..., 0] == rgb[..., 1]).all() and (rgb[..., 0] == rgb[..., 2]).all()


def test_minmax_constant_and_nan():
    assert (minmax_to_uint8(np.full((3, 3), 7.0)) == 0).all()
    out = minmax_to_uint8(np.array([0.0, np.nan, 10.0]))
    assert out.tolist() == [0, 0, 255]


def test_to_rgb_transform_is_picklable():
    assert isinstance(pickle.loads(pickle.dumps(ToRGB())), ToRGB)


@pytest.mark.parametrize(
    "name, label, expected",
    [
        ("IM-0115-0001.jpeg", "NORMAL", "IM-0115"),
        ("NORMAL2-IM-0123-0001.jpeg", "NORMAL", "NORMAL2-IM-0123"),
        ("IM-0011-0001-0002.jpeg", "NORMAL", "IM-0011"),  # extra suffix, same patient
        ("person1_bacteria_1.jpeg", "PNEUMONIA", "person1"),
        ("person1_virus_6.jpeg", "PNEUMONIA", "person1"),
        ("person447_virus_921_1.jpeg", "PNEUMONIA", "person447"),
        ("scan.jpeg", "NORMAL", None),
        ("person1_bacteria_1.jpeg", "NORMAL", None),  # pattern is per class
        ("IM-0115.jpeg", "NORMAL", None),
    ],
)
def test_patient_id(name, label, expected):
    assert patient_id(name, label) == expected


LAYOUT = {
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
    for split, classes in LAYOUT.items():
        for label, names in classes.items():
            folder = tmp_path / split / label
            folder.mkdir(parents=True)
            for name in names:
                Image.new("L", (40, 30), 100).save(folder / name, format="JPEG")
    return tmp_path


def test_scan_and_patient_overlap(tiny_dataset):
    records = scan_dataset(tiny_dataset)
    assert len(records) == 9 and all(r.patient for r in records)
    overlap = patient_overlap(records)
    assert overlap["train/val"]["shared_patients"] == 1
    assert overlap["train/val"]["val_images_from_shared_patients"] == 1
    assert overlap["train/test"]["shared_patients"] == 1
    assert overlap["train/test"]["test_images"] == 3
    assert overlap["val/test"]["shared_patients"] == 1


def test_fingerprint_changes_when_files_change(tiny_dataset):
    first = dataset_fingerprint(scan_dataset(tiny_dataset))
    assert first == dataset_fingerprint(scan_dataset(tiny_dataset))
    Image.new("L", (40, 30), 0).save(tiny_dataset / "test" / "NORMAL" / "IM-0009-0001.jpeg")
    assert dataset_fingerprint(scan_dataset(tiny_dataset)) != first


def test_image_folder_and_sample_patients(tiny_dataset):
    test = image_folder(tiny_dataset, "test")
    assert test.classes == ["NORMAL", "PNEUMONIA"]
    assert sample_patients(test) == ["IM-0003", "person1", "person3"]
    assert raw_image(test, 0).size == (40, 30)


def test_missing_split_and_bad_classes_raise(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset split not found"):
        image_folder(tmp_path, "train")
    (tmp_path / "train" / "CATS").mkdir(parents=True)
    Image.new("L", (8, 8)).save(tmp_path / "train" / "CATS" / "a.png")
    with pytest.raises(ValueError, match="expected"):
        image_folder(tmp_path, "train")
    with pytest.raises(FileNotFoundError, match="Expected folder"):
        scan_dataset(tmp_path)


def test_fake_split_has_both_classes_and_valid_tensors():
    dataset = split_dataset("unused", "val", build_eval_transform(224, 256), fake=True)
    assert len(dataset) == 8 and set(dataset.targets) == {0, 1}
    x, y = dataset[0]
    assert tuple(x.shape) == (3, 224, 224) and y == 0
    assert isinstance(raw_image(dataset, 1), Image.Image)
    assert sample_patients(dataset) == [None] * 8
    train = split_dataset("unused", "train", None, fake=True)
    assert not np.array_equal(np.asarray(train[0][0]), np.asarray(dataset.raw_image(0)))


def test_class_weights_are_balanced():
    weights = class_weights([0, 1, 1, 1])
    assert torch.allclose(weights, torch.tensor([2.0, 2.0 / 3.0]))
    with pytest.raises(ValueError, match="Every class"):
        class_weights([1, 1])


def test_balanced_sampler_draws_classes_equally():
    targets = [0] * 100 + [1] * 900
    sampler = balanced_sampler(targets, seed=0)
    drawn = Counter(targets[i] for _ in range(5) for i in sampler)
    assert abs(drawn[0] / sum(drawn.values()) - 0.5) < 0.03


def test_seeded_loader_is_reproducible():
    dataset = split_dataset("unused", "train", build_train_transform(224), fake=True)

    def first_batch():
        seed_everything(0)
        loader = make_loader(
            dataset, 4, num_workers=0, device=torch.device("cpu"), shuffle=True, seed=0
        )
        return next(iter(loader))

    (x1, y1), (x2, y2) = first_batch(), first_batch()
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
