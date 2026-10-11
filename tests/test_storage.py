# stage02_v2
"""Тесты storage: round-trip сохранения/загрузки + сканирование папки."""

import json

import numpy as np
import pytest

import keras
from keras import layers

from digit_app.storage import (
    ModelMeta,
    ModelEntry,
    save_model,
    load_model,
    list_models,
    delete_model,
)


def tiny_model():
    """Крошечная модель для тестов: 4 входа -> 2 выхода."""
    m = keras.Sequential([
        layers.Input(shape=(4,)),
        layers.Dense(2, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="mse")
    return m


def sample_meta(acc=0.9876):
    return ModelMeta(
        created_at="2026-10-11T11:22:33",
        accuracy=acc,
        loss=0.0234,
        epochs=3,
        batch_size=8,
        learning_rate=0.001,
        optimizer="adam",
        conv_filters=[4, 8, 8],
        dense_units=16,
    )


def test_save_creates_both_files(tmp_path):
    m = tiny_model()
    entry = save_model(m, sample_meta(), models_dir=tmp_path)
    assert entry.keras_path.exists()
    assert entry.keras_path.suffix == ".keras"
    json_path = entry.keras_path.with_suffix(".json")
    assert json_path.exists()


def test_save_json_content(tmp_path):
    m = tiny_model()
    meta = sample_meta(acc=0.5)
    entry = save_model(m, meta, models_dir=tmp_path)
    raw = json.loads(entry.keras_path.with_suffix(".json").read_text("utf-8"))
    assert raw["accuracy"] == 0.5
    assert raw["epochs"] == 3
    assert raw["optimizer"] == "adam"
    assert raw["conv_filters"] == [4, 8, 8]
    assert raw["model_file"] == entry.keras_path.name
    assert "app_version" in raw


def test_save_then_load_roundtrip(tmp_path):
    m = tiny_model()
    x = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    y_before = m.predict(x, verbose=0)

    entry = save_model(m, sample_meta(), models_dir=tmp_path)
    m2 = load_model(entry)
    y_after = m2.predict(x, verbose=0)

    assert np.allclose(y_before, y_after, atol=1e-5)


def test_filename_contains_accuracy(tmp_path):
    m = tiny_model()
    entry = save_model(m, sample_meta(acc=0.4321), models_dir=tmp_path)
    assert "0.4321" in entry.keras_path.name


def test_filename_starts_with_timestamp(tmp_path):
    m = tiny_model()
    entry = save_model(m, sample_meta(), models_dir=tmp_path)
    stem = entry.keras_path.stem
    parts = stem.split("_")
    assert len(parts[0]) == 8 and parts[0].isdigit()
    assert len(parts[1]) == 6 and parts[1].isdigit()


def test_collision_gets_suffix(tmp_path):
    m = tiny_model()
    meta = sample_meta()
    e1 = save_model(m, meta, models_dir=tmp_path)
    e2 = save_model(m, meta, models_dir=tmp_path)
    assert e1.keras_path != e2.keras_path
    assert e1.keras_path.exists() and e2.keras_path.exists()


def test_list_models_empty_dir(tmp_path):
    assert list_models(tmp_path) == []


def test_list_models_nonexistent_dir(tmp_path):
    assert list_models(tmp_path / "no_such_dir") == []


def test_list_models_returns_saved(tmp_path):
    m = tiny_model()
    save_model(m, sample_meta(acc=0.1), models_dir=tmp_path)
    save_model(m, sample_meta(acc=0.2), models_dir=tmp_path)
    entries = list_models(tmp_path)
    assert len(entries) == 2
    assert all(isinstance(e, ModelEntry) for e in entries)


def test_list_models_ignores_orphans(tmp_path):
    m = tiny_model()
    save_model(m, sample_meta(), models_dir=tmp_path)
    (tmp_path / "orphan.keras").write_bytes(b"garbage")
    entries = list_models(tmp_path)
    assert len(entries) == 1
    assert "orphan" not in [e.name for e in entries]


def test_list_models_ignores_broken_json(tmp_path):
    m = tiny_model()
    save_model(m, sample_meta(), models_dir=tmp_path)
    (tmp_path / "broken.keras").write_bytes(b"garbage")
    (tmp_path / "broken.json").write_text("not json {{", encoding="utf-8")
    entries = list_models(tmp_path)
    assert len(entries) == 1


def test_delete_model(tmp_path):
    m = tiny_model()
    entry = save_model(m, sample_meta(), models_dir=tmp_path)
    assert entry.keras_path.exists()
    delete_model(entry)
    assert not entry.keras_path.exists()
    assert not entry.keras_path.with_suffix(".json").exists()


def test_delete_model_idempotent(tmp_path):
    m = tiny_model()
    entry = save_model(m, sample_meta(), models_dir=tmp_path)
    delete_model(entry)
    delete_model(entry)  # повторно — не падает
