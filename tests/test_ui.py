# stage05_v2
"""Smoke-тесты скелета UI.

Один Tk-инстанс на модуль (module-scoped fixture) — на Windows
создание более ~20 Tk() подряд валит TclError. App пересоздаётся
per-test внутри одного root.
"""

import tkinter as tk
from tkinter import ttk

import pytest

from digit_app.ui import App, OPTIMIZERS


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
    # Чистим всё, что оставил предыдущий тест
    for child in list(tk_root.winfo_children()):
        try:
            child.destroy()
        except tk.TclError:
            pass
    # Заглушка messagebox — на всякий случай (в test_ui_train важнее)
    monkeypatch.setattr(
        "digit_app.ui.messagebox.showerror",
        lambda *a, **kw: None,
    )
    a = App(tk_root)
    yield a
    # Останавливаем возможные потоки
    a._stop_flag.set()
    for child in list(tk_root.winfo_children()):
        try:
            child.destroy()
        except tk.TclError:
            pass


def test_window_creates(app):
    assert app.root is not None
    assert "digit_app" in app.root.title()


def test_two_tabs(app):
    tabs = app.notebook.tabs()
    assert len(tabs) == 2
    labels = [app.notebook.tab(t, "text") for t in tabs]
    assert "Обучение" in labels
    assert "Тест" in labels


def test_train_defaults(app):
    assert app.epochs_var.get() == 10
    assert app.batch_size_var.get() == 128
    assert abs(app.lr_var.get() - 0.001) < 1e-12
    assert app.optimizer_var.get() == "adam"
    assert app.conv1_var.get() == 28
    assert app.conv2_var.get() == 56
    assert app.conv3_var.get() == 56
    assert app.dense_var.get() == 56


def test_train_buttons_exist(app):
    assert app.btn_train is not None
    assert app.btn_stop is not None
    assert str(app.btn_stop["state"]) == "disabled"
    assert str(app.btn_train["state"]) in ("normal", "active")


def test_train_status_default(app):
    assert app.status_var.get()


def test_train_has_chart(app):
    assert app.fig is not None
    assert app.ax is not None
    assert app.canvas is not None


def test_optimizer_values(app):
    combos = []

    def walk(w):
        for c in w.winfo_children():
            if isinstance(c, ttk.Combobox):
                combos.append(c)
            walk(c)
    walk(app.train_tab)
    assert len(combos) >= 1
    values = list(combos[0]["values"])
    assert values == list(OPTIMIZERS)


def test_test_canvas_size(app):
    w = int(app.draw_canvas["width"])
    h = int(app.draw_canvas["height"])
    assert w == 280
    assert h == 280


def test_test_preproc_default_B(app):
    assert app.preproc_var.get() == "B"


def test_test_buttons_exist(app):
    assert app.btn_clear is not None
    assert app.btn_predict is not None
    assert app.btn_hist is not None


def test_test_result_default_dash(app):
    assert app.result_var.get() == "—"


def test_test_listbox_empty_at_start(app):
    # «текущая» появится только после обучения, в этом тесте пусто
    assert app.model_listbox.size() == 0


def test_test_preview_exists(app):
    assert app.preview_canvas is not None
