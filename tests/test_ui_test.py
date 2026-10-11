# stage06
"""Тесты вкладки «Тест» и кнопки «Сохранить модель».

Модель не обучается — подставляем FakeModel, возвращающий заданные
вероятности. Рисование эмулируется вызовами _draw_segment и
программными событиями через _on_canvas_motion с фейковым event.
"""

import tkinter as tk
from types import SimpleNamespace

import numpy as np
import pytest

from digit_app.ui import App
from digit_app.train import TrainResult, TrainSettings


# ---------- fixtures ----------

@pytest.fixture(scope="module")
def tk_root():
    root = tk.Tk()
    root.withdraw()
    yield root
    try:
        root.quit()
    except tk.TclError:
        pass
    try:
        root.destroy()
    except tk.TclError:
        pass


@pytest.fixture
def app(tk_root, monkeypatch, tmp_path):
    for child in list(tk_root.winfo_children()):
        try:
            child.destroy()
        except tk.TclError:
            pass
    monkeypatch.setattr(
        "digit_app.ui.messagebox.showerror",
        lambda *a, **kw: None,
    )
    # models_dir — во временную папку
    monkeypatch.setattr(
        "digit_app.ui.list_models",
        lambda *a, **kw: [],
    )
    a = App(tk_root)
    a._stop_flag.set()
    yield a
    a._stop_flag.set()
    if a._train_thread is not None:
        a._train_thread.join(timeout=5.0)
    for child in list(tk_root.winfo_children()):
        try:
            child.destroy()
        except tk.TclError:
            pass


class FakeModel:
    """Возвращает заранее заданный вектор вероятностей."""

    def __init__(self, probs=None):
        self.probs = (np.array([0.01] * 10, dtype=np.float32)
                      if probs is None else np.asarray(probs, dtype=np.float32))
        self.predict_calls = 0

    def predict(self, x, verbose=0):
        self.predict_calls += 1
        return self.probs.reshape(1, -1)


def ev(x, y):
    return SimpleNamespace(x=x, y=y)


def simulate_stroke(app, points):
    """Проводит «мышкой» по списку точек."""
    if not points:
        return
    app._on_canvas_press(ev(*points[0]))
    for p in points[1:]:
        app._on_canvas_motion(ev(*p))
    app._on_canvas_release(ev(*points[-1]))


# ---------- рисование ----------

def test_canvas_image_starts_white(app):
    arr = np.asarray(app._canvas_image)
    assert arr.shape == (280, 280)
    assert arr.min() == 255


def test_draw_dot_marks_pixels(app):
    app._on_canvas_press(ev(140, 140))
    arr = np.asarray(app._canvas_image)
    assert arr[140, 140] == 0  # центр кисти — чёрный
    assert arr[0, 0] == 255    # угол — белый


def test_draw_segment_marks_pixels(app):
    app._on_canvas_press(ev(50, 140))
    app._on_canvas_motion(ev(230, 140))
    arr = np.asarray(app._canvas_image)
    # на линии
    assert arr[140, 140] == 0
    # вне линии далеко
    assert arr[10, 140] == 255


def test_clear_resets_everything(app):
    simulate_stroke(app, [(50, 50), (230, 230)])
    assert np.asarray(app._canvas_image).min() == 0
    app._on_clear_click()
    assert np.asarray(app._canvas_image).min() == 255
    assert app.result_var.get() == "—"


def test_simulated_stroke_produces_black(app):
    simulate_stroke(app, [(100, 100), (150, 100), (150, 150)])
    arr = np.asarray(app._canvas_image)
    assert arr.min() == 0
    black_ratio = float((arr < 128).mean())
    assert 0.001 < black_ratio < 0.5


# ---------- предсказание ----------

def test_predict_without_model_shows_hint(app):
    simulate_stroke(app, [(100, 100), (150, 150)])
    app._do_predict()
    assert "Выберите модель" in app.result_var.get()


def test_predict_with_model_shows_top3(app):
    probs = np.zeros(10, dtype=np.float32)
    probs[7] = 0.9
    probs[1] = 0.08
    probs[2] = 0.02
    app._active_model = FakeModel(probs)
    simulate_stroke(app, [(100, 100), (150, 150)])
    app._do_predict()
    text = app.result_var.get()
    assert "7" in text and "90" in text
    assert "1" in text and "8" in text
    assert "2" in text


def test_predict_shows_preview(app):
    app._active_model = FakeModel()
    simulate_stroke(app, [(100, 100), (150, 150)])
    app._do_predict()
    items = app.preview_canvas.find_all()
    assert len(items) >= 1
    assert app._preview_photo is not None


def test_predict_stores_last_probs(app):
    probs = np.arange(10, dtype=np.float32) + 1.0
    probs /= probs.sum()
    app._active_model = FakeModel(probs)
    simulate_stroke(app, [(100, 100), (150, 150)])
    app._do_predict()
    assert hasattr(app, "_last_probs")
    assert app._last_probs.shape == (10,)


def test_predict_normalizes_logits(app):
    """Если модель вернёт логиты — UI должен их нормализовать."""
    logits = np.array([-2.0] * 10, dtype=np.float32)
    logits[3] = 5.0
    app._active_model = FakeModel(logits)
    simulate_stroke(app, [(100, 100), (150, 150)])
    app._do_predict()
    total = float(app._last_probs.sum())
    assert abs(total - 1.0) < 1e-5
    top = int(np.argmax(app._last_probs))
    assert top == 3


def test_preproc_switch_changes_result(app):
    """A и B дают разные превью (или хотя бы разные массивы)."""
    simulate_stroke(app, [(80, 120), (200, 140)])
    app.preproc_var.set("A")
    arrA = app._preprocess_current()
    app.preproc_var.set("B")
    arrB = app._preprocess_current()
    assert arrA.shape == (28, 28)
    assert arrB.shape == (28, 28)
    assert not np.array_equal(arrA, arrB)


# ---------- дебаунс ----------

def test_motion_schedules_debounce(app):
    app._cancel_debounce()
    assert app._debounce_id is None
    app._on_canvas_press(ev(100, 100))
    app._on_canvas_motion(ev(150, 150))
    assert app._debounce_id is not None
    app._cancel_debounce()


def test_predict_click_cancels_debounce(app):
    app._on_canvas_press(ev(100, 100))
    app._on_canvas_motion(ev(150, 150))
    assert app._debounce_id is not None
    app._on_predict_click()
    assert app._debounce_id is None


# ---------- выбор модели ----------

def test_select_current_sets_active(app):
    fake = FakeModel()
    app._current_model = fake
    app._last_result = TrainResult(model=fake, history={},
                                   test_loss=0.1, test_accuracy=0.9)
    app.refresh_model_list()
    assert app._active_model is fake


def test_select_index_marks_selection(app):
    fake = FakeModel()
    app._current_model = fake
    app._last_result = TrainResult(model=fake, history={},
                                   test_loss=0.1, test_accuracy=0.9)
    app.refresh_model_list()
    app._select_index(0)
    sel = app.model_listbox.curselection()
    assert 0 in [int(i) for i in sel]


# ---------- кнопка «Сохранить модель» ----------

def test_save_button_disabled_initially(app):
    assert str(app.btn_save["state"]) == "disabled"


def test_save_button_enabled_after_training(app, monkeypatch):
    # имитируем финиш обучения
    fake_model = FakeModel()
    app._current_model = fake_model
    app._last_result = TrainResult(model=fake_model, history={},
                                   test_loss=0.1, test_accuracy=0.9)
    app._last_settings = TrainSettings()
    app._update_save_button_state()
    assert str(app.btn_save["state"]) in ("normal", "active")


def test_save_click_writes_files(app, monkeypatch, tmp_path):
    import keras
    from keras import layers

    real_model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ])
    real_model.compile(optimizer="adam",
                       loss="sparse_categorical_crossentropy")

    app._current_model = real_model
    app._last_result = TrainResult(model=real_model, history={},
                                   test_loss=0.1, test_accuracy=0.9)
    app._last_settings = TrainSettings()

    saved_entries = []

    def fake_save(model, meta, models_dir=None):
        # не пишем по-настоящему; просто фиксируем вызов
        saved_entries.append(meta)
        return SimpleNamespace(name="fake_entry", keras_path=None, meta=meta)

    monkeypatch.setattr("digit_app.ui.save_model", fake_save)
    app._on_save_click()
    assert len(saved_entries) == 1
    meta = saved_entries[0]
    assert meta.accuracy == 0.9
    assert meta.optimizer == "adam"
    assert meta.conv_filters == [28, 56, 56]


# ---------- гистограмма ----------

def test_histogram_creates_window(app):
    probs = np.zeros(10, dtype=np.float32)
    probs[4] = 1.0
    app._active_model = FakeModel(probs)
    simulate_stroke(app, [(100, 100), (150, 150)])
    app._do_predict()
    app._on_hist_click()
    assert app._hist_window is not None
    assert app._hist_window.winfo_exists()
    app._hist_window.destroy()
