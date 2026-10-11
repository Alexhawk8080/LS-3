# stage01
"""Тесты препроцессинга (режимы A и B)."""

import numpy as np
import pytest

from digit_app.preprocess import (
    preprocess_naive,
    preprocess_mnist_like,
    OUT_SIZE,
    INNER_SIZE,
)


def blank_canvas(h=280, w=280):
    return np.full((h, w), 255, dtype=np.uint8)


def draw_rect(canvas, y0, y1, x0, x1):
    out = canvas.copy()
    out[y0:y1, x0:x1] = 0
    return out


def center_of_mass(img):
    total = float(img.sum())
    if total == 0:
        return None
    ys, xs = np.indices(img.shape)
    return float((ys * img).sum() / total), float((xs * img).sum() / total)


def bbox(img, thresh=0.5):
    mask = img > thresh
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


@pytest.mark.parametrize("fn", [preprocess_naive, preprocess_mnist_like])
def test_shape_dtype_range(fn):
    canvas = draw_rect(blank_canvas(), 100, 180, 100, 180)
    out = fn(canvas)
    assert out.shape == (OUT_SIZE, OUT_SIZE)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_naive_inverts_white_bg():
    canvas = draw_rect(blank_canvas(), 100, 180, 100, 180)
    out = preprocess_naive(canvas)
    assert out[OUT_SIZE // 2, OUT_SIZE // 2] > 0.5
    assert out[0, 0] < 0.5


def test_mnist_like_empty_canvas_returns_zeros():
    out = preprocess_mnist_like(blank_canvas())
    assert out.shape == (OUT_SIZE, OUT_SIZE)
    assert out.max() == 0.0


def test_mnist_like_centers_by_mass():
    canvas = draw_rect(blank_canvas(), 200, 260, 200, 260)
    out = preprocess_mnist_like(canvas)
    com = center_of_mass(out)
    assert com is not None
    cy, cx = com
    target = (OUT_SIZE - 1) / 2.0
    assert abs(cy - target) <= 2.0, f"cy={cy} далеко от {target}"
    assert abs(cx - target) <= 2.0, f"cx={cx} далеко от {target}"


def test_mnist_like_scales_to_inner():
    canvas = draw_rect(blank_canvas(), 120, 160, 120, 160)
    out = preprocess_mnist_like(canvas)
    box = bbox(out)
    assert box is not None
    y0, y1, x0, x1 = box
    h = y1 - y0 + 1
    w = x1 - x0 + 1
    assert abs(max(h, w) - INNER_SIZE) <= 2


def test_mnist_like_preserves_aspect():
    canvas = draw_rect(blank_canvas(), 90, 190, 130, 150)
    out = preprocess_mnist_like(canvas)
    box = bbox(out)
    assert box is not None
    y0, y1, x0, x1 = box
    h = y1 - y0 + 1
    w = x1 - x0 + 1
    assert h > 2 * w, f"h={h}, w={w}: пропорции не сохранились"


def test_naive_does_not_crash_on_empty():
    out = preprocess_naive(blank_canvas())
    assert out.shape == (OUT_SIZE, OUT_SIZE)
    assert out.max() < 0.5
