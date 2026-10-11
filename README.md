<!-- stage07 -->
# digit_app — обучение и распознавание цифр MNIST

Десктопное приложение на Tkinter для:
1. Обучения свёрточной нейросети на MNIST — настройки через форму,
   прогресс в реальном времени, график loss/accuracy.
2. Сохранения моделей с метаданными (accuracy, epochs, optimizer, ...).
3. Тестирования: рисуешь цифру мышкой — модель предсказывает,
   показывает топ-3 и гистограмму по всем 10 классам.

## Установка

Требуется Python 3.10+ (проверено на 3.12).

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    pip install -r requirements-dev.txt

Если pip ругается на SSL (корпоративный прокси) — добавь
`--trusted-host pypi.org --trusted-host files.pythonhosted.org`.

## Запуск

Двойной клик по `start.bat` — без окна консоли.
Или из терминала для отладки:

    .venv\Scripts\python.exe main.py

## Как пользоваться

### Вкладка «Обучение»

1. Настрой параметры (по умолчанию — разумные: 10 эпох, batch 128,
   Adam, lr=0.001, фильтры 28/56/56).
2. Нажми «Обучить». Прогресс виден в прогресс-баре и на графике.
3. Кнопка «Стоп» останавливает обучение после текущей эпохи.
4. Когда готово — статус покажет test_accuracy на тестовой выборке.
5. Кнопка «Сохранить модель» запишет модель и сайдкар в `models/`.

### Вкладка «Тест»

1. Выбери модель из списка справа (сохранённые + «текущая»).
2. Нарисуй цифру мышкой в поле 280×280.
3. Через 300 мс после последнего движения — автоматическое предсказание.
   Или нажми «Распознать».
4. Слева — топ-3 с вероятностями и превью 28×28 (что реально
   увидела модель).
5. «Гистограмма» — отдельное окно со всеми 10 вероятностями.
6. Переключатель «Препроцессинг» — режимы A (наивный) и B
   (как в MNIST: bbox + центровка по центру масс). По умолчанию B.

## Что в моделях

Каждая сохранённая модель — это пара файлов в `models/`:

- `<YYYYMMDD_HHMMSS>_<accuracy>.keras` — сама модель (Keras 3).
- `<YYYYMMDD_HHMMSS>_<accuracy>.json` — метаданные:
  accuracy, loss, epochs, batch_size, learning_rate, optimizer,
  conv_filters, dense_units, created_at, app_version.

Осиротевшие файлы (только .keras или только .json) при сканировании
игнорируются — модель без метаданных не считается моделью.

## Тесты

    .venv\Scripts\python.exe -m pytest tests -v
    # или
    .venv\Scripts\python.exe run_tests.py

79 тестов: препроцессинг (A/B), storage (round-trip, сканирование),
обучение (настройки, сборка, fit, колбэки), UI (скелет, обучение через
UI, тест через UI, сохранение модели).

## Структура

    digit_app/
        __init__.py        # __version__
        preprocess.py      # preprocess_naive / preprocess_mnist_like
        storage.py         # ModelMeta, save/load/list/delete
        train.py           # TrainSettings, build_model, train_model
        ui.py              # App (Tkinter)
    tests/                 # pytest
    models/                # сохранённые модели (gitignored)
    main.py                # точка входа
    start.bat              # запуск без консоли
    run_tests.py           # обёртка pytest
    pytest.ini
    requirements.txt       # продуктовые зависимости
    requirements-dev.txt   # только pytest
    context.md, Plan.md    # документация

## Лицензия

Учебный проект. Используй как хочешь.
