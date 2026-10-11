<!-- stage07_docs -->
# digit_app — контекст

Краткий контекст проекта. Обновляется по мере продвижения.

## Что это

Десктопное приложение (Tkinter) для:
1. Обучения свёрточной нейросети на MNIST с настройками через форму.
2. Сохранения моделей в папке models вместе с JSON-сайдкаром.
3. Тестирования модели: пользователь рисует цифру мышкой,
   приложение препроцессит её и выдаёт предсказание.

## Стек

- Python 3.12, .venv в корне проекта.
- TensorFlow 2.21.0, Keras 3.15.1 (через TF), numpy 2.5.3.
- Tkinter (стандартная библиотека) + matplotlib 3.11.2.
- Pillow 12.3.0 — для препроцессинга и PIL-буфера рисования.
- pytest 8.3.3 — тесты.

## Структура

digit_app/                # пакет
    __init__.py           # __version__
    preprocess.py         # режимы A (наивный) и B (как в MNIST)
    storage.py            # save/load .keras + .json, сканирование models/
    train.py              # TrainSettings, build_model, train_model
    ui.py                 # Tkinter, App (две вкладки)
tests/                    # pytest (79 тестов)
models/                   # сохранённые модели (в .gitignore)
main.py                   # точка входа
start.bat                 # запуск без консоли
run_tests.py              # обёртка pytest для clipbridge
pytest.ini
requirements.txt          # исходный (UTF-16), не трогаем
requirements-dev.txt      # только pytest
_dev/                     # stage-скрипты (в .gitignore)

## Ключевые решения

- **Препроцессинг — два режима**: A (наивный resize) и B (bbox +
  масштаб до 20x20 + центровка по центру масс + инверсия). По умолчанию B.
- **Сохранение** — пара файлов <YYYYMMDD_HHMMSS>_<acc>.keras + .json.
  Осиротевшие файлы игнорируются.
- **Обучение** — в отдельном потоке, колбэки on_epoch_end и should_stop,
  общение через queue.Queue, опрос из UI раз в 100 мс.
- **Рисование** — параллельный PIL-буфер (ImageDraw), синхронный с canvas.
  Это позволяет получать пиксели без ImageGrab и тестировать без экрана.
- **Дебаунс предсказания** — 300 мс после последнего движения мыши.
- **Список моделей** — сохранённые + «★ текущая, не сохранена».
- **Канвас** 280x280, кисть 20 px (фиксированная).

## История под-этапов

| Этап | Что сделано | Тестов |
|------|-------------|--------|
| stage00 | окружение, pytest, trusted-host для pip | — |
| stage01 | пакет, preprocess (A/B) | 8 |
| stage02 | storage (save/load/scan) | 13 |
| stage03 | train (settings/build/fit/callbacks) | 16 |
| stage04 | скелет UI (две вкладки) | 13 |
| stage04_docs | context.md + Plan.md | — |
| stage05 | обучение через UI (поток, очередь, график, стоп) | 10 |
| stage06 | вкладка «Тест» + сохранение модели | 19 |
| stage07 | main.py, README.md, start.bat | — |
| stage07_docs | финальная синхронизация | — |

Всего тестов: 79 passed.

## Особенности окружения

- MITM-прокси требует trusted-host для pip (pip.ini в %APPDATA%).
- Кодировка stdout дочерних процессов — UTF-8.
- Путь проекта содержит пробел (LS 3) — все subprocess через список
  аргументов, не строку.
- На Windows больше ~20 Tk()-инстансов за один pytest-прогон валит
  TclError. В тестах UI используется module-scoped fixture (один Tk
  на файл).

## Известные мелочи

- Keras 3 при load_model ругается на несовпадение optimizer-переменных —
  это безопасно для inference, но если позже захочется дообучать
  сохранённую модель, надо будет пересобирать optimizer.
- DeprecationWarning из keras/src/backend/tensorflow/core.py про
  numpy 2.x copy — внутреннее, не наше.
