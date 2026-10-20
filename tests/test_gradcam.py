from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest
import torch
from PIL import ImageOps

from lungxray.gradcam import grad_cam, overlay


def _assert_valid_cam(cam):
    assert cam.shape == (224, 224)
    assert cam.dtype == np.float32
    assert np.isfinite(cam).all()
    assert cam.min() >= 0.0 and cam.max() <= 1.0


def test_cam_shape_and_range(predictor, sample_image):
    cam = predictor.explain(sample_image)
    _assert_valid_cam(cam)
    # A real map, not all zeros. The peak is normalized to 1 at feature resolution; after
    # bilinear upsampling it can land slightly below 1.
    assert cam.max() > 0.9
    assert cam.std() > 0.01


def test_constant_image_gives_no_nan(predictor, constant_image):
    _assert_valid_cam(predictor.explain(constant_image))


def test_no_hooks_and_no_param_grads(predictor, sample_image):
    predictor.explain(sample_image)
    predictor.explain(sample_image, target="NORMAL")
    for module in predictor.model.modules():
        assert not module._forward_hooks
        assert not module._forward_pre_hooks
        assert not module._backward_hooks
        assert not module._backward_pre_hooks
    assert all(p.grad is None for p in predictor.model.parameters())
    assert not predictor.model.training


def test_concurrent_calls_match_sequential(predictor, sample_image):
    images = [sample_image, ImageOps.mirror(sample_image), ImageOps.flip(sample_image)]
    sequential = [predictor.explain(img) for img in images]
    # The maps must differ per image, otherwise a cross-thread mix-up would go unnoticed.
    assert not np.allclose(sequential[0], sequential[1], atol=1e-3)
    assert not np.allclose(sequential[0], sequential[2], atol=1e-3)
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(predictor.explain, images * 3))
    for expected, got in zip(sequential * 3, concurrent, strict=True):
        np.testing.assert_allclose(got, expected, atol=1e-5)


def test_works_inside_no_grad(predictor, sample_image):
    x = predictor.preprocess(sample_image)
    with torch.no_grad():
        cam = grad_cam(predictor.model, x, target_class=1)
    _assert_valid_cam(cam)


def test_target_selection(predictor, sample_image):
    x = predictor.preprocess(sample_image)
    seen = []
    cam = grad_cam(predictor.model, x, target_class=lambda logits: seen.append(logits) or 0)
    _assert_valid_cam(cam)
    assert seen[0].shape == (2,) and not seen[0].requires_grad
    with pytest.raises(ValueError, match="out of range"):
        grad_cam(predictor.model, x, target_class=5)


def test_rejects_training_mode_and_unsupported_models(predictor, sample_image):
    x = predictor.preprocess(sample_image)
    predictor.model.train()
    try:
        with pytest.raises(ValueError, match="eval"):
            grad_cam(predictor.model, x)
    finally:
        predictor.model.eval()
    with pytest.raises(TypeError, match="features"):
        grad_cam(torch.nn.Linear(2, 2).eval(), x)
    with pytest.raises(ValueError, match="shape"):
        grad_cam(predictor.model, torch.cat([x, x]))


def test_overlay_colours_are_rgb():
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    hot = overlay(rgb, np.ones((10, 12), np.float32), alpha=1.0)
    cold = overlay(rgb, np.zeros((10, 12), np.float32), alpha=1.0)
    assert hot.dtype == np.uint8 and hot.shape == rgb.shape
    assert hot[0, 0, 0] > 100 and hot[0, 0, 2] == 0  # high CAM is red
    assert cold[0, 0, 2] > 100 and cold[0, 0, 0] == 0  # low CAM is blue


def test_overlay_resizes_cam_and_validates_inputs():
    rgb = np.full((50, 40, 3), 200, dtype=np.uint8)
    cam = np.zeros((7, 7), np.float32)
    cam[3, 3] = np.nan
    out = overlay(rgb, cam, alpha=0.0)
    np.testing.assert_array_equal(out, rgb)
    expected = cv2.addWeighted(rgb, 0.5, overlay(rgb, cam, alpha=1.0), 0.5, 0.0)
    np.testing.assert_array_equal(overlay(rgb, cam, alpha=0.5), expected)
    with pytest.raises(ValueError, match="alpha"):
        overlay(rgb, cam, alpha=1.5)
    with pytest.raises(ValueError, match="uint8"):
        overlay(rgb.astype(np.float32), cam)
