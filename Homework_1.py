import tensorflow as tf
import tensorflow_datasets as tfds

# 1. Загрузка
(ds_train, ds_test), info = tfds.load(
    "cifar10",
    split=["train", "test"],
    with_info=True,
    as_supervised=True,
)

# 2. Препроцессинг
def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

BATCH_SIZE = 128
ds_train = ds_train.map(normalize).cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_test  = ds_test.map(normalize).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 3. Модель
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 4. Обучение
model.fit(ds_train, epochs=5, validation_data=ds_test)