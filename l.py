import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Загружаем данные
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images, test_images = train_images/255.0, test_images/255.0

model=models.Sequential([
    layers.Conv2D(28, (4, 4), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(56, (4, 4), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(56, (4, 4), activation="relu"),
    layers.Flatten(),
    layers.Dense(56, activation="relu"),
    layers.Dense(10, activation="softmax"),
    ])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
history=model.fit(
    train_images, train_labels,
    epochs=10,
    validation_data=(test_images,test_labels)
)

class_names = ['0','1','2','3','4','5','6','7','8','9']
 