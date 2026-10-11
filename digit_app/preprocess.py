# stage01
"""Препроцессинг изображения цифры в формат MNIST.

Соглашение о входе
------------------
image : 2D numpy array (H, W) со значениями 0..255 (uint8) или
        PIL.Image в режиме "L".
        Фон СВЕТЛЫЙ (высокие значения), штрих ТЁМНЫЙ (низкие значения).
        Соответствует рисованию тёмной кистью по белому холсту.

Соглашение о выходе
-------------------
numpy array (28, 28), dtype=float32, значения 0..1.
Фон НИЗКИЙ (близко к 0), штрих ВЫСОКИЙ (близко к 1).
Соответствует формату, который ожидает модель, обученная на MNIST.

Режимы
------
A (naive)      : весь холст масштабируется до 28x28 без анализа содержимого.
B (mnist_like) : цифра вырезается по bounding box, масштабируется до 20x20
                 с сохранением пропорций, помещается в 28x28 с центровкой
                 по центру масс. Это воспроизводит процедуру создания MNIST
                 (LeCun et al., 1998).
"""

import numpy as np
from PIL import Image


OUT_SIZE = 28
INNER_SIZE = 20
GRAY_THRESHOLD = 128


def _to_gray_array(image):
    """Приводит вход к 2D numpy uint8 array (H, W)."""
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("L"))
    else:
        image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(
            f"ожидается 2D массив, получено shape={image.shape}"
        )
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _resize(arr, size_hw):
    """Ресайз 2D uint8 массива через PIL (LANCZOS). size_hw = (h, w)."""
    img = Image.fromarray(arr, mode="L")
    return np.asarray(img.resize((size_hw[1], size_hw[0]), Image.LANCZOS))


def preprocess_naive(image):
    """Режим A: простое масштабирование всего холста до 28x28.

    Без вырезания, без центровки. Инверсия делается — иначе модель,
    обученная на MNIST, получит инвертированные данные и не сможет
    работать. Это намеренно «плохой» baseline для сравнения с B.
    """
    arr = _to_gray_array(image)
    resized = _resize(arr, (OUT_SIZE, OUT_SIZE))
    inverted = 255 - resized
    return inverted.astype(np.float32) / 255.0


def preprocess_mnist_like(image):
    """Режим B: как в MNIST — bbox, масштаб, центровка по центру масс."""
    arr = _to_gray_array(image)
    strokes = arr < GRAY_THRESHOLD

    if not strokes.any():
        return np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.float32)

    ys, xs = np.where(strokes)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cropped = arr[y0:y1, x0:x1]

    h, w = cropped.shape
    scale = INNER_SIZE / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = _resize(cropped, (new_h, new_w))

    inverted = (255 - resized).astype(np.float64)

    total = inverted.sum()
    if total == 0:
        return np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.float32)

    ys_i, xs_i = np.indices(inverted.shape)
    cy = float((ys_i * inverted).sum() / total)
    cx = float((xs_i * inverted).sum() / total)

    target = (OUT_SIZE - 1) / 2.0
    top = int(round(target - cy))
    left = int(round(target - cx))

    canvas = np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.float64)

    dest_y0 = max(0, top)
    dest_x0 = max(0, left)
    dest_y1 = min(OUT_SIZE, top + new_h)
    dest_x1 = min(OUT_SIZE, left + new_w)

    src_y0 = dest_y0 - top
    src_x0 = dest_x0 - left
    src_y1 = src_y0 + (dest_y1 - dest_y0)
    src_x1 = src_x0 + (dest_x1 - dest_x0)

    canvas[dest_y0:dest_y1, dest_x0:dest_x1] = (
        inverted[src_y0:src_y1, src_x0:src_x1]
    )

    return canvas.astype(np.float32) / 255.0
