from keras import layers, models
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Загрузка набора данных MNIST (Изображения рукописных цифр).
(train_images, train_labels), (val_images, val_labels) = mnist.load_data()

# Приводим значения пикселей до диапазона от 0 до 1 и изменяем форму данных так,
# чтобы они соответствовали входу сверточной нейронной сети 28x28 пикселей с каналом 1.
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
val_images = val_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

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
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

# Настройка оптимизатора и выбор функции потерь
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Подготовка данных для обучения с использованием генератора расширения данных
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
datagen.fit(train_images)

# Обучение модели
batch_size = 64
epochs = 6
steps_per_epoch = len(train_images) / batch_size

history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=batch_size),
    steps_per_epoch=steps_per_epoch, epochs=epochs,
    validation_data=(val_images, val_labels)
)

# Сохранение обученной модели
model.save('cnn_model.keras')

# Построение графиков
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# график изменения точности на обучающей и валидационной выборке
plt.plot(acc, label='Обучающая')
plt.plot(val_acc, label='Валидационная')
plt.title('Точность на выборках')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()

plt.figure()

# график изменения потери на обучающей и валидационной выборке
plt.plot(loss, label='Обучающая')
plt.plot(val_loss, label='Валидационная')
plt.title('Потери на выборках')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()

plt.show()