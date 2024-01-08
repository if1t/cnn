import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

# Загрузка набора данных MNIST (Изображения рукописных цифр от 0 до 9).
(train_images, train_labels), (val_images, val_labels) = mnist.load_data()

# Приводим значения пикселей до диапазона от 0 до 1 и изменяем форму данных так,
# чтобы они соответствовали входу сверточной нейронной сети 28x28 пикселей с каналом 1.
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

val_images = val_images.reshape((10000, 28, 28, 1))
val_images = val_images.astype('float32') / 255

# Кодирование меток в векторы
train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)

# Конструирование сверточной нейронной сети
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Добавление полносвязных слоев
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Настройка оптимизатора и выбор функции потерь
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Подготовка данных для обучения с использованием генератора расширения данных
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
datagen.fit(train_images)

# Обучение модели
batch_size = 32
history = model.fit(datagen.flow(train_images, train_labels, batch_size), steps_per_epoch=len(train_images) / batch_size, epochs=10, validation_data=(val_images, val_labels))

# Сохранение обученной модели
model.save('cnn_model.keras')

# Загрузка сохраненной модели
model = load_model('cnn_model.keras')

# Оценка производительности на тестовых данных
test_loss, test_accuracy = model.evaluate(val_images, val_labels)

# Вывод результатов
print('Точность на тестовой выборке:', test_accuracy)
