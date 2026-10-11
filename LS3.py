# python 3.8+
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ------------------------------------------------------------
# 1. Данные: атмосферное давление в атм -> PSI
#    1 атм = 14.6959 PSI
# ------------------------------------------------------------
atm = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0], dtype=float)
psi = atm * 14.6959

print("Обучающие примеры:")
for a, p in zip(atm, psi):
    print(f"{a:.2f} атм = {p:.4f} PSI")

# ------------------------------------------------------------
# 2. Модель с несколькими скрытыми слоями
#    Вход: 1 признак (атм)
#    Скрытые слои: 16 -> 16 -> 8 нейронов, активация ReLU
#    Выход: 1 нейрон (линейная активация)
# ------------------------------------------------------------
model = Sequential([
    Dense(32, activation='relu', input_dim=1),
    Dense(32, activation='relu'),
    Dense(16,  activation='relu'),
    Dense(1)  # линейный выход для регрессии
])

# Компиляция: MSE + Adam с уменьшенным шагом (0.01)
model.compile(loss='mean_squared_error', optimizer=Adam(0.01))

# Обучение: 1000 эпох, без вывода лога
history = model.fit(atm, psi, epochs=1000, verbose=0)

# ------------------------------------------------------------
# 3. Проверка на новых значениях
# ------------------------------------------------------------
test_atm = np.array([1.0, 1.5, 2.2, 3.7])
predictions = model.predict(test_atm)

print("\nРезультаты предсказания:")
for a, p in zip(test_atm, predictions):
    true_val = a * 14.6959
    print(f"{a:.2f} атм -> {p[0]:.4f} PSI  (истина: {true_val:.4f})")

# ------------------------------------------------------------
# 4. Архитектура и веса
# ------------------------------------------------------------
print("\nСтруктура модели:")
model.summary()

# ------------------------------------------------------------
# 5. График потерь
# ------------------------------------------------------------
plt.plot(history.history['loss'], color='blue')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Обучение модели: атм -> PSI (много слоёв)")
plt.grid(True)
plt.show()