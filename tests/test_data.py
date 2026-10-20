import pickle

import numpy as np
import pytest
from PIL import Image

from lungxray.data import ToRGB, build_eval_transform, minmax_to_uint8, to_rgb


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
