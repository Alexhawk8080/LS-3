# stage02_v2
"""Сохранение, загрузка и сканирование моделей digit_app.

Работаем с парой файлов в models/:
  <name>.keras — модель (keras 3 / TF 2.x)
  <name>.json  — метаданные (настройки обучения, метрики, дата)
"""

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

import keras  # noqa: F401

from digit_app import __version__


DEFAULT_MODELS_DIR = Path("models")


@dataclass
class ModelMeta:
    """Метаданные обученной модели (всё, что лежит в .json).

    Поле model_file заполняется во время save_model() — до сохранения
    оно пустое, потому что имя файла формируется из времени и accuracy.
    """

    created_at: str
    accuracy: float
    loss: float
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    conv_filters: list
    dense_units: int
    model_file: str = ""
    app_version: str = field(default_factory=lambda: __version__)

    @classmethod
    def from_dict(cls, d):
        # Игнорируем неизвестные поля — на будущее, если схема расширится.
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ModelEntry:
    """Пара «модель + метаданные», как видит их UI."""

    name: str
    keras_path: Path
    meta: ModelMeta


def _timestamp_name(accuracy: float) -> str:
    """Имя файла: <YYYYMMDD_HHMMSS>_<acc>. Время локальное."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{accuracy:.4f}"


def save_model(model, meta: ModelMeta,
               models_dir=DEFAULT_MODELS_DIR) -> ModelEntry:
    """Сохраняет модель и сайдкар. Имя файла генерируется из времени и accuracy.

    Возвращает ModelEntry. Если папки нет — создаёт.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    base = _timestamp_name(meta.accuracy)
    keras_path = models_dir / f"{base}.keras"
    json_path = models_dir / f"{base}.json"

    # Гарантируем, что не затираем (маловероятно при секундной точности,
    # но на всякий случай — добавляем суффикс).
    i = 0
    while keras_path.exists() or json_path.exists():
        i += 1
        keras_path = models_dir / f"{base}_{i}.keras"
        json_path = models_dir / f"{base}_{i}.json"

    model.save(keras_path)

    # Обновляем model_file и записываем сайдкар.
    meta_dict = asdict(meta)
    meta_dict["model_file"] = keras_path.name
    json_path.write_text(
        json.dumps(meta_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    meta_out = ModelMeta.from_dict(meta_dict)
    return ModelEntry(name=keras_path.stem, keras_path=keras_path, meta=meta_out)


def load_model(entry: ModelEntry):
    """Загружает модель по ModelEntry."""
    return keras.models.load_model(entry.keras_path)


def list_models(models_dir=DEFAULT_MODELS_DIR) -> list:
    """Сканирует models/ и возвращает список ModelEntry.

    Пара считается валидной, если есть и .keras, и .json.
    Осиротевшие файлы (.keras без .json или наоборот) игнорируются.
    Сортировка: по created_at, свежие сверху.
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []

    entries = []
    for keras_path in sorted(models_dir.glob("*.keras")):
        json_path = keras_path.with_suffix(".json")
        if not json_path.exists():
            continue
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            meta = ModelMeta.from_dict(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        entries.append(ModelEntry(
            name=keras_path.stem,
            keras_path=keras_path,
            meta=meta,
        ))

    entries.sort(key=lambda e: e.meta.created_at, reverse=True)
    return entries


def delete_model(entry: ModelEntry) -> None:
    """Удаляет модель и её сайдкар. Не падает, если файлов уже нет."""
    for p in (entry.keras_path, entry.keras_path.with_suffix(".json")):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
