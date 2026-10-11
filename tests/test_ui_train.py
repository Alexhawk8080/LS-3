# stage05_v2
"""Тесты обучения через UI. Один Tk на модуль (см. test_ui.py).
train_model подменяется быстрым фейком.
"""

import threading
import time
import tkinter as tk

import pytest

from digit_app.ui import App
from digit_app.train import TrainResult


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
def app(tk_root, monkeypatch):
    for child in list(tk_root.winfo_children()):
        try:
            child.destroy()
        except tk.TclError:
            pass
    errors = []
    monkeypatch.setattr(
        "digit_app.ui.messagebox.showerror",
        lambda title, msg, **kw: errors.append((title, msg)),
    )
    a = App(tk_root)
    a._test_errors = errors
    yield a
    # Гарантированно останавливаем потоки перед уничтожением
    a._stop_flag.set()
    if a._train_thread is not None:
        a._train_thread.join(timeout=5.0)
    for child in list(tk_root.winfo_children()):
        try:
            child.destroy()
        except tk.TclError:
            pass


def wait_for(predicate, timeout=5.0, step=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return False


def drain_until_idle(app, timeout=5.0):
    if app._train_thread is not None:
        app._train_thread.join(timeout=timeout)
    app._drain_queue()
    app.root.update_idletasks()


def fake_train_factory(epochs_to_emit=2, stop_after=None):
    calls = {"settings": None, "epochs_called": 0}

    def fake_train(settings, *, on_epoch_end=None, should_stop=None, **kw):
        calls["settings"] = settings
        history = {"loss": [], "accuracy": [],
                   "val_loss": [], "val_accuracy": []}
        for e in range(epochs_to_emit):
            if should_stop is not None and should_stop():
                break
            logs = {
                "loss": 1.0 - 0.1 * e,
                "accuracy": 0.5 + 0.1 * e,
                "val_loss": 0.9 - 0.05 * e,
                "val_accuracy": 0.55 + 0.1 * e,
            }
            history["loss"].append(logs["loss"])
            history["accuracy"].append(logs["accuracy"])
            history["val_loss"].append(logs["val_loss"])
            history["val_accuracy"].append(logs["val_accuracy"])
            if on_epoch_end is not None:
                on_epoch_end(e, logs)
            calls["epochs_called"] = e + 1
            if stop_after is not None and e >= stop_after:
                break
        return TrainResult(
            model="FAKE_MODEL",
            history=history,
            test_loss=0.12,
            test_accuracy=0.87,
        )

    return fake_train, calls


# ---------- _read_settings ----------

def test_read_settings_from_form(app):
    app.epochs_var.set(3)
    app.batch_size_var.set(32)
    app.lr_var.set(0.01)
    app.optimizer_var.set("sgd")
    app.conv1_var.set(8)
    app.conv2_var.set(16)
    app.conv3_var.set(16)
    app.dense_var.set(24)
    s = app._read_settings()
    assert s.epochs == 3
    assert s.batch_size == 32
    assert abs(s.learning_rate - 0.01) < 1e-12
    assert s.optimizer == "sgd"
    assert s.conv_filters == [8, 16, 16]
    assert s.dense_units == 24


def test_read_settings_invalid_raises(app):
    app.epochs_var.set(0)
    with pytest.raises(ValueError):
        app._read_settings()


def test_train_click_shows_error_on_invalid(app, monkeypatch):
    app.epochs_var.set(0)
    app._on_train_click()
    assert len(app._test_errors) == 1
    assert app._train_thread is None


def test_train_click_runs_to_completion(app, monkeypatch):
    fake_train, calls = fake_train_factory(epochs_to_emit=3)
    monkeypatch.setattr("digit_app.ui.train_model", fake_train)

    app.epochs_var.set(3)
    app._on_train_click()

    assert wait_for(lambda: app._train_thread is not None
                    and not app._train_thread.is_alive())
    drain_until_idle(app)

    assert calls["epochs_called"] == 3
    assert app._current_model == "FAKE_MODEL"
    assert "Готово" in app.status_var.get()
    assert "0.87" in app.status_var.get()


def test_progress_advances(app, monkeypatch):
    fake_train, _ = fake_train_factory(epochs_to_emit=3)
    monkeypatch.setattr("digit_app.ui.train_model", fake_train)

    app.epochs_var.set(3)
    app._on_train_click()
    wait_for(lambda: app._train_thread is not None
             and not app._train_thread.is_alive())
    drain_until_idle(app)

    assert int(app.progress["value"]) == 3
    assert int(app.progress["maximum"]) == 3


def test_chart_has_lines_after_training(app, monkeypatch):
    fake_train, _ = fake_train_factory(epochs_to_emit=2)
    monkeypatch.setattr("digit_app.ui.train_model", fake_train)

    app.epochs_var.set(2)
    app._on_train_click()
    wait_for(lambda: app._train_thread is not None
             and not app._train_thread.is_alive())
    drain_until_idle(app)

    assert len(app.ax.lines) >= 4


def test_buttons_state_after_training(app, monkeypatch):
    fake_train, _ = fake_train_factory(epochs_to_emit=1)
    monkeypatch.setattr("digit_app.ui.train_model", fake_train)

    app.epochs_var.set(1)
    app._on_train_click()
    wait_for(lambda: app._train_thread is not None
             and not app._train_thread.is_alive())
    drain_until_idle(app)

    assert str(app.btn_train["state"]) in ("normal", "active")
    assert str(app.btn_stop["state"]) == "disabled"


def test_model_list_shows_current_after_training(app, monkeypatch):
    fake_train, _ = fake_train_factory(epochs_to_emit=1)
    monkeypatch.setattr("digit_app.ui.train_model", fake_train)

    app.epochs_var.set(1)
    app._on_train_click()
    wait_for(lambda: app._train_thread is not None
             and not app._train_thread.is_alive())
    drain_until_idle(app)

    assert app.model_listbox.size() >= 1
    assert "текущая" in app.model_listbox.get(0)


def test_stop_sets_flag(app, monkeypatch):
    started = threading.Event()

    def blocking_train(settings, *, on_epoch_end=None, should_stop=None, **kw):
        started.set()
        for _ in range(60):
            if should_stop is not None and should_stop():
                break
            time.sleep(0.05)
        return TrainResult(model="X", history={}, test_loss=0.0, test_accuracy=0.0)

    monkeypatch.setattr("digit_app.ui.train_model", blocking_train)
    app.epochs_var.set(5)
    app._on_train_click()
    assert started.wait(timeout=3.0)

    app._on_stop_click()
    assert app._stop_flag.is_set()

    wait_for(lambda: app._train_thread is not None
             and not app._train_thread.is_alive(), timeout=8.0)
    drain_until_idle(app, timeout=8.0)


def test_error_in_worker_shows_message(app, monkeypatch):
    def raising_train(settings, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr("digit_app.ui.train_model", raising_train)

    app.epochs_var.set(1)
    app._on_train_click()
    wait_for(lambda: app._train_thread is not None
             and not app._train_thread.is_alive())
    drain_until_idle(app)

    assert len(app._test_errors) == 1
    assert "boom" in app._test_errors[0][1]
    assert "Ошибка" in app.status_var.get()
    assert str(app.btn_train["state"]) in ("normal", "active")
    assert str(app.btn_stop["state"]) == "disabled"
