# stage03
"""Обучение модели распознавания цифр.

UI не импортируется. Всё взаимодействие — через on_epoch_end/should_stop.
"""

from dataclasses import dataclass, field

import keras
from keras import layers


OPTIMIZERS = ("adam", "sgd", "rmsprop")


@dataclass
class TrainSettings:
    """Настройки обучения. Соответствуют полям формы на вкладке «Обучение»."""

    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 0.001
    optimizer: str = "adam"
    conv_filters: list = field(default_factory=lambda: [28, 56, 56])
    dense_units: int = 56

    def __post_init__(self):
        if self.optimizer.lower() not in OPTIMIZERS:
            raise ValueError(
                f"оптимизатор {self.optimizer!r} не поддерживается; "
                f"выбери из {OPTIMIZERS}"
            )
        if len(self.conv_filters) != 3:
            raise ValueError(
                f"conv_filters должен содержать ровно 3 значения, "
                f"получено {len(self.conv_filters)}"
            )
        if any(int(f) < 1 for f in self.conv_filters):
            raise ValueError("все conv_filters должны быть >= 1")
        if self.epochs < 1:
            raise ValueError("epochs >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size >= 1")
        if self.dense_units < 1:
            raise ValueError("dense_units >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate > 0")


@dataclass
class TrainResult:
    """Результат обучения."""

    model: object
    history: dict
    test_loss: float
    test_accuracy: float


def _make_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr)
    raise ValueError(f"неизвестный оптимизатор: {name}")


def build_model(settings: TrainSettings,
                input_shape=(28, 28, 1),
                num_classes: int = 10):
    """Собирает и компилирует модель по настройкам.

    Архитектура повторяет исходный l.py, но с осью канала и без
    конфликта softmax/from_logits.
    """
    f1, f2, f3 = (int(x) for x in settings.conv_filters)
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(f1, (4, 4), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(f2, (4, 4), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(f3, (4, 4), activation="relu"),
        layers.Flatten(),
        layers.Dense(int(settings.dense_units), activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=_make_optimizer(settings.optimizer, settings.learning_rate),
        loss="sparse_categorical_crossentropy",  # from_logits=False
        metrics=["accuracy"],
    )
    return model


class _ProgressCallback(keras.callbacks.Callback):
    """Пробрасывает наружу on_epoch_end и проверяет should_stop."""

    def __init__(self, on_epoch_end=None, should_stop=None):
        super().__init__()
        self._on_epoch_end = on_epoch_end
        self._should_stop = should_stop

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self._on_epoch_end is not None:
            self._on_epoch_end(epoch, dict(logs))
        if self._should_stop is not None and self._should_stop():
            self.model.stop_training = True


def load_mnist():
    """MNIST с осью канала и нормализацией в 0..1.

    Возвращает ((x_train, y_train), (x_test, y_test)):
      x_*: float32 (N, 28, 28, 1), значения 0..1
      y_*: uint8 (N,)
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train[..., None].astype("float32")) / 255.0
    x_test = (x_test[..., None].astype("float32")) / 255.0
    return (x_train, y_train), (x_test, y_test)


def train_model(settings: TrainSettings, *,
                train_data=None, test_data=None,
                on_epoch_end=None, should_stop=None,
                verbose=0) -> TrainResult:
    """Обучает модель.

    train_data/test_data — кортежи (x, y). Если не переданы — грузится MNIST.
    on_epoch_end(epoch, logs) — вызывается после каждой эпохи (epoch 0-based).
    should_stop() -> bool — если True, обучение останавливается после текущей.

    Возвращает TrainResult.
    """
    if train_data is None or test_data is None:
        (x_train, y_train), (x_test, y_test) = load_mnist()
    else:
        (x_train, y_train) = train_data
        (x_test, y_test) = test_data

    model = build_model(settings)
    cb = _ProgressCallback(on_epoch_end=on_epoch_end, should_stop=should_stop)
    history = model.fit(
        x_train, y_train,
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        validation_data=(x_test, y_test),
        callbacks=[cb],
        verbose=verbose,
    )
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    return TrainResult(
        model=model,
        history={k: list(v) for k, v in history.history.items()},
        test_loss=float(loss),
        test_accuracy=float(acc),
    )
