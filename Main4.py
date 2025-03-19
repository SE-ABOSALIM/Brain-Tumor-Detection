import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Dataset yolunu ayarlayın
train_dir = './cleaned-Dataset/Training'
test_dir = './cleaned-Dataset/Testing'

# Görsellerin boyutu
img_size = 150
batch_size = 32

# Veri artırma (augmentation) işlemleri ile ImageDataGenerator oluştur
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Görüntüleri normalleştir
    rotation_range=20,         # Rastgele döndürme
    width_shift_range=0.2,     # Yatayda kaydırma
    height_shift_range=0.2,    # Dikeyde kaydırma
    shear_range=0.2,           # Kesme işlemi
    zoom_range=0.2,            # Yakınlaştırma
    horizontal_flip=True,      # Yatayda çevirme
    fill_mode='nearest'        # Eksik pikselleri doldurma
)

# Test verisi için sadece normalizasyon yapıyoruz
test_datagen = ImageDataGenerator(rescale=1./255)

# Eğitim verilerini yükle
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'  # Çok sınıflı sınıflandırma
)

# Test verilerini yükle
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

# CNN modelini oluştur
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 sınıf olduğu için 4 çıktı ve softmax aktivasyonu
])

# Öğrenme hızını ayarlayın
optimizer = Adam(learning_rate=0.0001)

# Modeli derle
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping ile overfitting'i önle
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Modeli eğit
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator,
    callbacks=[early_stopping]
)

# Eğitim ve doğrulama sonuçlarını çizdir
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Modeli değerlendirme
loss, accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {accuracy * 100:.2f}%")
