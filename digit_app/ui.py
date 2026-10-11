# stage06
"""Tkinter-интерфейс digit_app. Полный.

Обучение — в отдельном потоке через queue.Queue.
Тест — рисование на canvas с параллельным PIL-буфером (для получения
пикселей без ImageGrab), автопредсказание с дебаунсом, выбор модели.
"""

import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from digit_app.train import TrainSettings, TrainResult, train_model
from digit_app.storage import ModelMeta, ModelEntry, list_models, load_model, save_model
from digit_app.preprocess import preprocess_naive, preprocess_mnist_like


WINDOW_TITLE = "digit_app — распознавание цифр"
WINDOW_MIN = (900, 650)

CANVAS_SIZE = 280
BRUSH_SIZE = 20
PREVIEW_SIZE = 84  # 28 * 3

OPTIMIZERS = ("adam", "sgd", "rmsprop")

POLL_INTERVAL_MS = 100
DEBOUNCE_MS = 300


class App:
    """Главное окно."""

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(WINDOW_TITLE)
        root.minsize(*WINDOW_MIN)

        # --- состояние обучения ---
        self._train_queue = queue.Queue()
        self._stop_flag = threading.Event()
        self._train_thread = None
        self._after_id = None
        self._history = {"loss": [], "accuracy": [],
                         "val_loss": [], "val_accuracy": []}
        self._current_model = None
        self._last_result = None
        self._last_settings = None

        # --- состояние теста ---
        self._canvas_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        self._canvas_draw = ImageDraw.Draw(self._canvas_image)
        self._last_point = None
        self._debounce_id = None
        self._preview_photo = None
        self._model_items = []
        self._active_model = None
        self._active_entry = None
        self._loaded_models = {}
        self._hist_window = None

        # --- UI ---
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)
        self.train_tab = ttk.Frame(self.notebook)
        self.test_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.train_tab, text="Обучение")
        self.notebook.add(self.test_tab, text="Тест")

        self._build_train_tab()
        self._build_test_tab()

        # --- команды кнопок ---
        self.btn_train.config(command=self._on_train_click)
        self.btn_stop.config(command=self._on_stop_click)
        self.btn_save.config(command=self._on_save_click)
        self.btn_clear.config(command=self._on_clear_click)
        self.btn_predict.config(command=self._on_predict_click)
        self.btn_hist.config(command=self._on_hist_click)

        # --- события canvas и listbox ---
        self.draw_canvas.bind("<Button-1>", self._on_canvas_press)
        self.draw_canvas.bind("<B1-Motion>", self._on_canvas_motion)
        self.draw_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.model_listbox.bind("<<ListboxSelect>>", self._on_model_select)

        self.refresh_model_list()
        self._schedule_poll()
        self._update_save_button_state()

    # ============================================================
    # Вкладка «Обучение»
    # ============================================================

    def _build_train_tab(self):
        parent = self.train_tab

        form = ttk.LabelFrame(parent, text="Настройки")
        form.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self.epochs_var = tk.IntVar(value=10)
        self.batch_size_var = tk.IntVar(value=128)
        self.lr_var = tk.DoubleVar(value=0.001)
        self.optimizer_var = tk.StringVar(value="adam")
        self.conv1_var = tk.IntVar(value=28)
        self.conv2_var = tk.IntVar(value=56)
        self.conv3_var = tk.IntVar(value=56)
        self.dense_var = tk.IntVar(value=56)

        r = 0
        ttk.Label(form, text="Эпохи:").grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Spinbox(form, from_=1, to=100, textvariable=self.epochs_var,
                    width=8).grid(row=r, column=1, sticky="w", padx=4, pady=2); r += 1
        ttk.Label(form, text="Батч:").grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Spinbox(form, from_=1, to=1024, textvariable=self.batch_size_var,
                    width=8).grid(row=r, column=1, sticky="w", padx=4, pady=2); r += 1
        ttk.Label(form, text="Learning rate:").grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(form, textvariable=self.lr_var, width=10).grid(
            row=r, column=1, sticky="w", padx=4, pady=2); r += 1
        ttk.Label(form, text="Оптимизатор:").grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Combobox(form, textvariable=self.optimizer_var, values=OPTIMIZERS,
                     state="readonly", width=8).grid(
            row=r, column=1, sticky="w", padx=4, pady=2); r += 1
        ttk.Separator(form, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1
        ttk.Label(form, text="Conv-слои (фильтры):").grid(
            row=r, column=0, columnspan=2, sticky="w", padx=4); r += 1
        ttk.Label(form, text="1:").grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Spinbox(form, from_=1, to=256, textvariable=self.conv1_var,
                    width=8).grid(row=r, column=1, sticky="w", padx=4, pady=2); r += 1
        ttk.Label(form, text="2:").grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Spinbox(form, from_=1, to=256, textvariable=self.conv2_var,
                    width=8).grid(row=r, column=1, sticky="w", padx=4, pady=2); r += 1
        ttk.Label(form, text="3:").grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Spinbox(form, from_=1, to=256, textvariable=self.conv3_var,
                    width=8).grid(row=r, column=1, sticky="w", padx=4, pady=2); r += 1
        ttk.Label(form, text="Dense-нейроны:").grid(
            row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Spinbox(form, from_=1, to=1024, textvariable=self.dense_var,
                    width=8).grid(row=r, column=1, sticky="w", padx=4, pady=2); r += 1
        ttk.Separator(form, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1
        self.btn_train = ttk.Button(form, text="Обучить")
        self.btn_train.grid(row=r, column=0, sticky="ew", padx=4, pady=2)
        self.btn_stop = ttk.Button(form, text="Стоп", state="disabled")
        self.btn_stop.grid(row=r, column=1, sticky="ew", padx=4, pady=2); r += 1
        self.btn_save = ttk.Button(form, text="Сохранить модель", state="disabled")
        self.btn_save.grid(row=r, column=0, columnspan=2, sticky="ew", padx=4, pady=2); r += 1
        self.progress = ttk.Progressbar(form, mode="determinate", maximum=100)
        self.progress.grid(row=r, column=0, columnspan=2, sticky="ew", padx=4, pady=4); r += 1
        self.status_var = tk.StringVar(value="Готово к запуску.")
        ttk.Label(form, textvariable=self.status_var, wraplength=260,
                  foreground="#555", justify="left").grid(
            row=r, column=0, columnspan=2, sticky="w", padx=4, pady=2)

        chart_frame = ttk.LabelFrame(parent, text="Loss / Accuracy по эпохам")
        chart_frame.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self._reset_chart()
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

    def _reset_chart(self):
        self.ax.clear()
        self.ax.set_xlabel("Эпоха")
        self.ax.set_ylabel("Значение")
        self.ax.grid(True, alpha=0.3)

    # ============================================================
    # Вкладка «Тест»
    # ============================================================

    def _build_test_tab(self):
        parent = self.test_tab

        left = ttk.Frame(parent)
        left.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        ttk.Label(left, text="Нарисуйте цифру мышкой:").pack(anchor="w")
        self.draw_canvas = tk.Canvas(
            left, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="white", highlightthickness=1,
            highlightbackground="#888",
        )
        self.draw_canvas.pack(pady=4)

        row = ttk.Frame(left)
        row.pack(fill="x")
        self.btn_clear = ttk.Button(row, text="Очистить")
        self.btn_clear.pack(side="left", padx=2)
        self.btn_predict = ttk.Button(row, text="Распознать")
        self.btn_predict.pack(side="left", padx=2)
        self.btn_hist = ttk.Button(row, text="Гистограмма")
        self.btn_hist.pack(side="left", padx=2)

        self.result_var = tk.StringVar(value="—")
        ttk.Label(left, text="Топ-3:").pack(anchor="w", pady=(8, 0))
        ttk.Label(left, textvariable=self.result_var,
                  font=("Consolas", 11), justify="left").pack(anchor="w")

        ttk.Label(left, text="Препроцессинг:").pack(anchor="w", pady=(8, 0))
        self.preproc_var = tk.StringVar(value="B")
        pre = ttk.Frame(left)
        pre.pack(anchor="w")
        ttk.Radiobutton(pre, text="A (наивный)", value="A",
                        variable=self.preproc_var).pack(side="left", padx=2)
        ttk.Radiobutton(pre, text="B (как в MNIST)", value="B",
                        variable=self.preproc_var).pack(side="left", padx=2)

        ttk.Label(left, text="Превью 28×28:").pack(anchor="w", pady=(8, 0))
        self.preview_canvas = tk.Canvas(
            left, width=PREVIEW_SIZE, height=PREVIEW_SIZE,
            bg="black", highlightthickness=1,
            highlightbackground="#888",
        )
        self.preview_canvas.pack(anchor="w")

        right = ttk.LabelFrame(parent, text="Модели")
        right.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)

        self.model_listbox = tk.Listbox(right, width=44, height=20,
                                        exportselection=False)
        self.model_listbox.pack(fill="both", expand=True, padx=4, pady=4)

        btn_row = ttk.Frame(right)
        btn_row.pack(fill="x", padx=4, pady=(0, 4))
        self.btn_reload_models = ttk.Button(
            btn_row, text="Обновить",
            command=self.refresh_model_list,
        )
        self.btn_reload_models.pack(side="left", padx=2)

        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

    # ============================================================
    # Список моделей и выбор
    # ============================================================

    def refresh_model_list(self):
        """Перечитывает models/ + «текущая», если есть.

        Заполняет self._model_items — параллельный список кортежей
        (kind, payload), где kind in {"current", "saved"}.
        """
        self.model_listbox.delete(0, tk.END)
        self._model_items = []

        if self._current_model is not None:
            label = "★ (текущая, не сохранена)"
            if self._last_result is not None:
                label += f"  acc={self._last_result.test_accuracy:.4f}"
            self.model_listbox.insert(tk.END, label)
            self.model_listbox.itemconfig(0, foreground="#1a7f37")
            self._model_items.append(("current", None))

        entries = list_models()
        for entry in entries:
            label = (f"{entry.name}  acc={entry.meta.accuracy:.4f}  "
                     f"ep={entry.meta.epochs}")
            self.model_listbox.insert(tk.END, label)
            self._model_items.append(("saved", entry))

        # если есть «текущая» — выбираем её сразу (удобно после обучения)
        if self._current_model is not None:
            self._select_index(0)

    def _select_index(self, i: int):
        """Программно выбирает элемент и проставляет активную модель."""
        if not (0 <= i < len(self._model_items)):
            return
        self.model_listbox.selection_clear(0, tk.END)
        self.model_listbox.selection_set(i)
        self._activate_item(i)

    def _on_model_select(self, _event=None):
        sel = self.model_listbox.curselection()
        if not sel:
            return
        self._activate_item(int(sel[0]))

    def _activate_item(self, i: int):
        kind, payload = self._model_items[i]
        if kind == "current":
            self._active_model = self._current_model
            self._active_entry = None
        else:
            entry: ModelEntry = payload
            if entry.name not in self._loaded_models:
                self.status_var.set(f"Загружаю {entry.name}...")
                try:
                    self._loaded_models[entry.name] = load_model(entry)
                except Exception as e:
                    messagebox.showerror("Модель",
                                         f"не удалось загрузить {entry.name}: {e}")
                    return
            self._active_model = self._loaded_models[entry.name]
            self._active_entry = entry

    # ============================================================
    # Рисование
    # ============================================================

    def _draw_segment(self, x0, y0, x1, y1):
        """Рисует отрезок и на canvas, и в PIL-буфере.

        На canvas — с capstyle=round (Tkinter), в PIL — линия + круги
        по концам (у ImageDraw.line нет capstyle, только joint).
        """
        self.draw_canvas.create_line(
            x0, y0, x1, y1,
            fill="black", width=BRUSH_SIZE, capstyle=tk.ROUND,
        )
        self._canvas_draw.line([(x0, y0), (x1, y1)], fill=0, width=BRUSH_SIZE)
        r = BRUSH_SIZE / 2
        for (cx, cy) in ((x0, y0), (x1, y1)):
            self._canvas_draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r], fill=0
            )

    def _draw_dot(self, x, y):
        r = BRUSH_SIZE / 2
        self.draw_canvas.create_oval(
            x - r, y - r, x + r, y + r, fill="black", outline="black"
        )
        self._canvas_draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def _on_canvas_press(self, event):
        self._last_point = (event.x, event.y)
        self._draw_dot(event.x, event.y)

    def _on_canvas_motion(self, event):
        if self._last_point is None:
            self._last_point = (event.x, event.y)
            self._draw_dot(event.x, event.y)
            return
        x0, y0 = self._last_point
        self._draw_segment(x0, y0, event.x, event.y)
        self._last_point = (event.x, event.y)
        self._schedule_debounce()

    def _on_canvas_release(self, _event):
        self._last_point = None
        self._schedule_debounce()

    def _on_clear_click(self):
        self._cancel_debounce()
        self.draw_canvas.delete("all")
        self._canvas_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        self._canvas_draw = ImageDraw.Draw(self._canvas_image)
        self._last_point = None
        self.result_var.set("—")
        self._clear_preview()

    def _clear_preview(self):
        self.preview_canvas.delete("all")
        self._preview_photo = None

    def _schedule_debounce(self):
        self._cancel_debounce()
        self._debounce_id = self.root.after(DEBOUNCE_MS, self._do_predict)

    def _cancel_debounce(self):
        if self._debounce_id is not None:
            try:
                self.root.after_cancel(self._debounce_id)
            except tk.TclError:
                pass
            self._debounce_id = None

    # ============================================================
    # Предсказание
    # ============================================================

    def _on_predict_click(self):
        self._cancel_debounce()
        self._do_predict()

    def _preprocess_current(self):
        """Возвращает (arr_28x28 float32, PIL-Image для превью)."""
        arr = np.asarray(self._canvas_image)  # (280, 280) uint8, white bg
        if self.preproc_var.get() == "A":
            out = preprocess_naive(arr)
        else:
            out = preprocess_mnist_like(arr)
        return out

    def _show_preview(self, arr_28):
        """arr_28 — float32 (28, 28) в 0..1."""
        img = Image.fromarray((arr_28 * 255.0).astype(np.uint8), mode="L")
        img = img.resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.NEAREST)
        self._preview_photo = ImageTk.PhotoImage(img)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(
            PREVIEW_SIZE // 2, PREVIEW_SIZE // 2, image=self._preview_photo
        )

    def _do_predict(self):
        self._debounce_id = None
        if self._active_model is None:
            self.result_var.set("Выберите модель в списке справа")
            return

        arr_28 = self._preprocess_current()
        self._show_preview(arr_28)

        x = arr_28.reshape(1, 28, 28, 1).astype(np.float32)
        try:
            probs = self._active_model.predict(x, verbose=0)[0]
        except Exception as e:
            self.result_var.set(f"Ошибка предсказания: {e}")
            return
        probs = np.asarray(probs, dtype=np.float64)
        # защита от логитов (если модель вернёт не softmax)
        if probs.min() < 0.0 or abs(float(probs.sum()) - 1.0) > 1e-3:
            e = np.exp(probs - probs.max())
            probs = e / e.sum()

        self._last_probs = probs
        top = np.argsort(-probs)[:3]
        lines = [f"{int(k)} — {probs[k] * 100:5.1f}%" for k in top]
        self.result_var.set("\n".join(lines))

    def _on_hist_click(self):
        if not hasattr(self, "_last_probs") or self._last_probs is None:
            # сначала нужно предсказание
            self._do_predict()
        if not hasattr(self, "_last_probs") or self._last_probs is None:
            return

        if self._hist_window is not None and self._hist_window.winfo_exists():
            self._hist_window.destroy()

        win = tk.Toplevel(self.root)
        win.title("Гистограмма вероятностей")
        self._hist_window = win

        fig = Figure(figsize=(5, 3.5), dpi=100)
        ax = fig.add_subplot(111)
        ax.bar(list(range(10)), self._last_probs, color="tab:blue")
        ax.set_xticks(list(range(10)))
        ax.set_xlabel("Цифра")
        ax.set_ylabel("Вероятность")
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        self._hist_fig = fig
        self._hist_canvas = canvas

    # ============================================================
    # Сохранение модели
    # ============================================================

    def _update_save_button_state(self):
        can_save = (self._current_model is not None
                    and self._last_result is not None
                    and self._last_settings is not None)
        self.btn_save.config(state=("normal" if can_save else "disabled"))

    def _on_save_click(self):
        if self._current_model is None or self._last_result is None:
            return
        s = self._last_settings
        r = self._last_result
        meta = ModelMeta(
            created_at=datetime.now().isoformat(timespec="seconds"),
            accuracy=float(r.test_accuracy),
            loss=float(r.test_loss),
            epochs=int(s.epochs),
            batch_size=int(s.batch_size),
            learning_rate=float(s.learning_rate),
            optimizer=str(s.optimizer),
            conv_filters=list(s.conv_filters),
            dense_units=int(s.dense_units),
        )
        try:
            entry = save_model(self._current_model, meta)
        except Exception as e:
            messagebox.showerror("Сохранение", f"не удалось сохранить: {e}")
            return
        self.status_var.set(f"Сохранено: {entry.name}")
        self.refresh_model_list()

    # ============================================================
    # Обучение
    # ============================================================

    def _read_settings(self) -> TrainSettings:
        try:
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            lr = float(self.lr_var.get())
            optimizer = str(self.optimizer_var.get())
            conv_filters = [
                int(self.conv1_var.get()),
                int(self.conv2_var.get()),
                int(self.conv3_var.get()),
            ]
            dense = int(self.dense_var.get())
        except (tk.TclError, ValueError) as e:
            raise ValueError(f"не удалось прочитать поле формы: {e}")

        return TrainSettings(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            optimizer=optimizer,
            conv_filters=conv_filters,
            dense_units=dense,
        )

    def _on_train_click(self):
        if self._train_thread is not None and self._train_thread.is_alive():
            return
        try:
            settings = self._read_settings()
        except ValueError as e:
            messagebox.showerror("Настройки", str(e))
            return
        self._start_training(settings)

    def _start_training(self, settings: TrainSettings):
        self._stop_flag.clear()
        self.btn_train.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_save.config(state="disabled")
        self.progress["value"] = 0
        self.progress["maximum"] = settings.epochs
        self.status_var.set("Обучение...")
        self._history = {"loss": [], "accuracy": [],
                         "val_loss": [], "val_accuracy": []}
        self._reset_chart()
        self.canvas.draw_idle()
        self._last_settings = settings

        def on_epoch(epoch, logs):
            self._train_queue.put(("epoch", epoch, logs))

        def should_stop():
            return self._stop_flag.is_set()

        def worker():
            try:
                result = train_model(
                    settings,
                    on_epoch_end=on_epoch,
                    should_stop=should_stop,
                )
                self._train_queue.put(("done", result))
            except Exception as e:
                self._train_queue.put(("error", e))

        self._train_thread = threading.Thread(target=worker, daemon=True)
        self._train_thread.start()

    def _on_stop_click(self):
        if self._train_thread is not None and self._train_thread.is_alive():
            self._stop_flag.set()
            self.status_var.set("Останавливаю после текущей эпохи...")

    # ============================================================
    # Опрос очереди
    # ============================================================

    def _schedule_poll(self):
        self._after_id = self.root.after(POLL_INTERVAL_MS, self._poll_queue)

    def _poll_queue(self):
        self._drain_queue()
        self._schedule_poll()

    def _drain_queue(self):
        while True:
            try:
                msg = self._train_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_message(msg)

    def _handle_message(self, msg):
        kind = msg[0]
        if kind == "epoch":
            _, epoch, logs = msg
            self.progress["value"] = epoch + 1
            self._history["loss"].append(logs.get("loss", 0.0))
            self._history["accuracy"].append(logs.get("accuracy", 0.0))
            self._history["val_loss"].append(logs.get("val_loss", 0.0))
            self._history["val_accuracy"].append(logs.get("val_accuracy", 0.0))
            self._update_chart()
            acc = logs.get("accuracy", 0.0)
            vacc = logs.get("val_accuracy", 0.0)
            self.status_var.set(
                f"Эпоха {epoch + 1}: acc={acc:.4f}  val_acc={vacc:.4f}"
            )
        elif kind == "done":
            _, result = msg
            self._on_train_finished(result)
        elif kind == "error":
            _, err = msg
            self._on_train_error(err)

    def _on_train_finished(self, result: TrainResult):
        self._current_model = result.model
        self._last_result = result
        self.btn_train.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status_var.set(
            f"Готово: test_acc={result.test_accuracy:.4f}  "
            f"test_loss={result.test_loss:.4f}"
        )
        self._update_chart()
        self.refresh_model_list()
        self._update_save_button_state()

    def _on_train_error(self, err: Exception):
        self.btn_train.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status_var.set(f"Ошибка: {err}")
        messagebox.showerror("Ошибка обучения", str(err))
        self._update_save_button_state()

    def _update_chart(self):
        h = self._history
        self.ax.clear()
        epochs = list(range(1, len(h["loss"]) + 1))
        if epochs:
            self.ax.plot(epochs, h["loss"], label="loss", color="tab:red")
            self.ax.plot(epochs, h["val_loss"], label="val_loss",
                         color="tab:orange", linestyle="--")
            self.ax.plot(epochs, h["accuracy"], label="accuracy",
                         color="tab:blue")
            self.ax.plot(epochs, h["val_accuracy"], label="val_accuracy",
                         color="tab:cyan", linestyle="--")
            self.ax.legend(fontsize=8)
        self.ax.set_xlabel("Эпоха")
        self.ax.set_ylabel("Значение")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
