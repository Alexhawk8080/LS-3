# stage03
"""Тесты train.py: настройки, сборка модели, обучение на синтетике.

Реальный MNIST грузится ровно в одном тесте — остальные используют
синтетические данные, чтобы не тратить время.
"""

import numpy as np
import pytest

from digit_app.train import (
    TrainSettings,
    TrainResult,
    build_model,
    train_model,
    load_mnist,
)


def tiny_settings(**overrides):
    base = dict(epochs=1, batch_size=16, learning_rate=0.01,
                optimizer="adam", conv_filters=[4, 4, 4], dense_units=8)
    base.update(overrides)
    return TrainSettings(**base)


def fake_mnist(n_train=64, n_test=32, seed=0):
    rng = np.random.default_rng(seed)
    x_train = rng.random((n_train, 28, 28, 1), dtype=np.float32)
    y_train = rng.integers(0, 10, n_train).astype(np.uint8)
    x_test = rng.random((n_test, 28, 28, 1), dtype=np.float32)
    y_test = rng.integers(0, 10, n_test).astype(np.uint8)
    return (x_train, y_train), (x_test, y_test)


# ---------- настройки ----------

def test_settings_defaults():
    s = TrainSettings()
    assert s.epochs == 10
    assert s.batch_size == 128
    assert s.optimizer == "adam"
    assert s.conv_filters == [28, 56, 56]
    assert s.dense_units == 56


def test_settings_rejects_bad_optimizer():
    with pytest.raises(ValueError, match="оптимизатор"):
        TrainSettings(optimizer="nesterov")


def test_settings_rejects_bad_conv_filters_count():
    with pytest.raises(ValueError, match="3 значения"):
        TrainSettings(conv_filters=[8, 16])


def test_settings_rejects_bad_epochs():
    with pytest.raises(ValueError, match="epochs"):
        TrainSettings(epochs=0)


def test_settings_rejects_bad_learning_rate():
    with pytest.raises(ValueError, match="learning_rate"):
        TrainSettings(learning_rate=0)


@pytest.mark.parametrize("name", ["adam", "sgd", "rmsprop"])
def test_settings_accepts_known_optimizers(name):
    s = TrainSettings(optimizer=name)
    assert s.optimizer == name


# ---------- сборка модели ----------

def test_build_model_output_shape():
    model = build_model(tiny_settings())
    x = np.zeros((1, 28, 28, 1), dtype=np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (1, 10)


def test_build_model_output_sums_to_one():
    """Проверяет, что выход — softmax-вероятности, а не логиты."""
    model = build_model(tiny_settings())
    x = np.random.default_rng(1).random((4, 28, 28, 1), dtype=np.float32)
    y = model.predict(x, verbose=0)
    sums = y.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-4)


# ---------- обучение ----------

def test_train_smoke():
    data = fake_mnist()
    result = train_model(tiny_settings(), train_data=data[0], test_data=data[1])
    assert isinstance(result, TrainResult)
    assert len(result.history["loss"]) == 1
    assert 0.0 <= result.test_accuracy <= 1.0
    assert result.test_loss >= 0.0


def test_train_calls_on_epoch_end():
    data = fake_mnist()
    epochs_seen = []
    train_model(tiny_settings(epochs=3),
                train_data=data[0], test_data=data[1],
                on_epoch_end=lambda e, logs: epochs_seen.append(e))
    assert epochs_seen == [0, 1, 2]


def test_train_logs_have_metrics():
    data = fake_mnist()
    seen = []
    train_model(tiny_settings(),
                train_data=data[0], test_data=data[1],
                on_epoch_end=lambda e, logs: seen.append(logs))
    logs = seen[0]
    assert "loss" in logs
    assert "accuracy" in logs
    assert "val_loss" in logs
    assert "val_accuracy" in logs


def test_train_stops_early():
    data = fake_mnist()
    settings = tiny_settings(epochs=10)
    result = train_model(settings,
                         train_data=data[0], test_data=data[1],
                         should_stop=lambda: True)
    # первая эпоха завершается, потом останавливаемся
    assert len(result.history["loss"]) == 1


def test_train_does_not_stop_without_flag():
    data = fake_mnist()
    settings = tiny_settings(epochs=3)
    result = train_model(settings,
                         train_data=data[0], test_data=data[1])
    assert len(result.history["loss"]) == 3


# ---------- MNIST (медленный) ----------

def test_load_mnist_shapes():
    (x_train, y_train), (x_test, y_test) = load_mnist()
    assert x_train.shape == (60000, 28, 28, 1)
    assert x_test.shape == (10000, 28, 28, 1)
    assert x_train.dtype == np.float32
    assert 0.0 <= float(x_train.min()) and float(x_train.max()) <= 1.0
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
