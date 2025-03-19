import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from PIL import Image
from warnings import filterwarnings

filterwarnings('ignore')

# Veri setini yükleme ve ön işleme
x_train = []  # MRI görüntüleri
y_train = []  # Görüntü etiketleri
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
image_size = 256

# Eğitim ve test veri setlerini yükleme
for i in labels:
    folderPath = os.path.join('C:\\Users\\90538\\Documents\\GitHub\\Brain-Tumor-Detection\\Dataset\\Training', i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        x_train.append(img)
        y_train.append(i)

for i in labels:
    folderPath = os.path.join('C:\\Users\\90538\\Documents\\GitHub\\Brain-Tumor-Detection\\Dataset\\Testing', i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        x_train.append(img)
        y_train.append(i)

x_train = np.array(x_train)
y_train = np.array(y_train)

# Etiket dağılımını görselleştirme
label_counts = {label: np.sum(y_train == label) for label in labels}
plt.figure(figsize=(8, 6))
colors = ["C0", "C1", "C2", "C3"]
bars = plt.bar(label_counts.keys(), label_counts.values(), color=colors)
plt.xlabel('Labels')
plt.ylabel('Count')
plt.title('Distribution of Labels')
plt.show()

# Etiketleri one-hot encoding formatına dönüştürme
y_train = np.array(pd.get_dummies(y_train))

# Veri setini eğitim, test ve doğrulama setlerine ayırma
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Piksel değerlerini normalize etme
x_train = x_train / 255
x_test = x_test / 255
x_valid = x_valid / 255

# Model tanımlama
model = Sequential()
model.add(Input(shape=(256, 256, 3)))
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.15))
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.15))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.15))
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.45))
model.add(Dense(4, activation='softmax'))

# Modeli derleme
model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Modeli eğitme
start_time = time.time()
history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=45, verbose=1, batch_size=64)
end_time = time.time()
runtime = end_time - start_time
print("Total runtime:", runtime, "seconds")

# Model doğruluk ve kayıp grafikleri
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Karışıklık matrisi ve sınıflandırma raporu
y_true = np.argmax(y_train, axis=1)
y_pred = np.argmax(model.predict(x_train), axis=1)
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

y_true_test = np.argmax(y_test, axis=1)
y_pred_test = np.argmax(model.predict(x_test), axis=1)
sns.heatmap(confusion_matrix(y_true_test, y_pred_test), annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_true_test, y_pred_test, target_names=labels))

# Rastgele bir görüntü üzerinde tahmin yapma
random_index = np.random.randint(0, len(x_test))
random_img = x_test[random_index]
predictions = model.predict(random_img.reshape(1, 256, 256, 3))
predicted_class = np.argmax(predictions)
predicted_label = labels[predicted_class]
confidence = predictions[0][predicted_class]
actual_index = y_test[random_index]
actual_class = np.argmax(actual_index)
actual_label = labels[actual_class]

print(f"Predicted label: {predicted_label} \nActual label: {actual_label} \nConfidence: {confidence*100:.2f}%\n")
plt.figure(figsize=(3, 3))
plt.imshow(random_img)
plt.axis('off')
plt.show()

# Özel bir görüntü üzerinde tahmin yapma
img_path = '/kaggle/input/test-dataset/no_tumor.jpg'
custom_actual_label = 'no_tumor'
custom_img_arr = []
tumor_img = cv2.imread(img_path)
tumor_img = cv2.resize(tumor_img, (256, 256))
custom_img_arr.append(tumor_img)
custom_img_arr = np.array(custom_img_arr)
custom_img_arr = custom_img_arr / 255

custom_pred = model.predict(custom_img_arr.reshape(1, 256, 256, 3))
custom_pred_class = np.argmax(custom_pred)
custom_pred_label = labels[custom_pred_class]
custom_pred_confidence = custom_pred[0][custom_pred_class]

print(f"Predicted label: {custom_pred_label} \nActual label: {custom_actual_label} \nConfidence: {custom_pred_confidence*100:.2f}%\n")
plt.figure(figsize=(3, 3))
plt.imshow(tumor_img)
plt.axis('off')
plt.show()