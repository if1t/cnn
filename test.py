from keras.preprocessing import image
from keras.models import load_model
import numpy as np

# Загрузка модели
model = load_model('cnn_model.keras')

# Обработка изображения из интернета
img = image.load_img('assets/5.jpg', target_size=(28, 28), color_mode='grayscale')
img_array = image.img_to_array(img)  # Преобразование изображения в массив numpy
img_array = img_array / 255.0  # Нормализация значений пикселей как в обучающей выборке
img_array = np.expand_dims(img_array, axis=0)  # Добавление измерения батча, так как модели ожидают массив батчей

# Использование модели
predicted_digit = np.argmax(model.predict(img_array))
print("Распознанное число:", predicted_digit)