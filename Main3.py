# Gerekli kütüphanelerin yüklenmesi
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Veri yolları (Kaggle'dan indirdiğiniz datasetin yolunu girin)
train_dir = './cleaned-Dataset/Training'  # Örneğin: 'data/Training'
test_dir = './cleaned-Dataset/Testing'    # Örneğin: 'data/Testing'

# Veri artırma ve ön işleme
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,  # Daha fazla veri artırma
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Validation için %20 ayır
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Eğitim ve validation veri setleri
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Test veri seti
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# ResNet50 modelini yükle (önceden eğitilmiş ağırlıklarla)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Modelin üzerine yeni katmanlar ekle
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.3)(x)  # Dropout oranını %30 yaptık
x = Dense(512, activation='relu')(x)  # Yeni bir katman ekledik
x = Dropout(0.3)(x)  # Yeni dropout
predictions = Dense(4, activation='softmax')(x)  # 4 sınıf için

# Modeli oluştur
model = Model(inputs=base_model.input, outputs=predictions)

# Daha derin katmanları eğitime aç
for layer in base_model.layers[:-4]:  # Sadece son 4 katmanı eğitime aç
    layer.trainable = False

# Modeli derle
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# Model özeti
model.summary()

# Callback'lerin tanımlanması
checkpoint = ModelCheckpoint('brain_tumor_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)

callbacks = [checkpoint, early_stopping, reduce_lr]

# Modeli eğit
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)

# Eğitim sonuçlarının görselleştirilmesi
plt.figure(figsize=(12, 4))

# Kayıp grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Doğruluk grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Test verisi üzerinde değerlendirme
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Tahminler ve karışıklık matrisi
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Karışıklık matrisi
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Sınıflandırma raporu
print(classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))
