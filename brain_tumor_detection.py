'''
FOR GOOGLE COLAB:
!git clone https://github.com/SE-ABOSALIM/Brain-Tumor-Detection.git
!ls Brain-Tumor-Detection
!ls -R Brain-Tumor-Detection/cleaned-Dataset/

FOR VSCODE:
!git clone https://github.com/SE-ABOSALIM/Brain-Tumor-Detection.git
!dir Brain-Tumor-Detection
!dir /S "Brain-Tumor-Detection\cleaned-Dataset"
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import itertools
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

np.random.seed(0)

dataset_path = "Brain-Tumor-Detection/Dataset"
train_path = os.path.join(dataset_path, "Training")
test_path = os.path.join(dataset_path, "Testing")
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
image_size = 150

def process_image(img_path):
    img = cv2.imread(img_path, 0)
    img = cv2.bilateralFilter(img, 2, 50, 50)
    img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    img = cv2.resize(img, (image_size, image_size))
    return img

x_train, y_train, x_test, y_test = [], [], [], []

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

for label in labels:
    category_index = labels.index(label)

    train_folder = os.path.join(train_path, label)
    for file in tqdm(os.listdir(train_folder), desc=f"Training - {label}"):
        img = cv2.imread(os.path.join(train_folder, file), 0)
        img = cv2.bilateralFilter(img, 2, 50, 50)
        img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
        img = cv2.resize(img, (image_size, image_size))
        x_train.append(img)
        y_train.append(category_index)

    test_folder = os.path.join(test_path, label)
    for file in tqdm(os.listdir(test_folder), desc=f"Testing - {label}"):
        img = cv2.imread(os.path.join(test_folder, file), 0)
        img = cv2.bilateralFilter(img, 2, 50, 50)
        img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
        img = cv2.resize(img, (image_size, image_size))
        x_test.append(img)
        y_test.append(category_index)

x_train = np.array(x_train) / 255.0
x_test = np.array(x_test) / 255.0

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

y_train = to_categorical(y_train, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)

img_size = (150, 150)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

print(x_train.shape)
print(x_test.shape)

fig, axes = plt.subplots(10, 7, figsize=(10, 10))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(x_train[i])
    ax.axis('off')
plt.tight_layout()
plt.show()

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(image_size, image_size, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
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

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.45))

    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
model.summary()

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True
)
datagen.fit(x_train)

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    validation_data=validation_generator,
    epochs=45,
    steps_per_epoch=len(x_train) // 32
)

score = model.evaluate(x_train, y_train, verbose=0)
train_loss = score[0] * 100
train_accuracy = score[1] * 100

print(f"Training Loss: {train_loss:.2f}%")
print(f"Training Accuracy: {train_accuracy:.2f}%")

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

random_index = np.random.randint(0, len(x_test))
random_img = x_test[random_index]

random_img_resized = cv2.resize(random_img, (150, 150))

random_img_resized = random_img_resized / 255.0

predictions = model.predict(random_img_resized.reshape(1, 150, 150, 3))
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

random_index = np.random.randint(0, len(x_test))
random_img = x_test[random_index]

random_img_resized = cv2.resize(random_img, (150, 150))
random_img_resized = random_img_resized / 255.0

predictions = model.predict(random_img_resized.reshape(1, 150, 150, 3))
predicted_class = np.argmax(predictions)
predicted_label = labels[predicted_class]
confidence = predictions[0][predicted_class]

actual_index = np.argmax(y_test[random_index])
actual_label = labels[actual_index]

print(f"Predicted label: {predicted_label} \nActual label: {actual_label} \nConfidence: {confidence*100:.2f}%\n")
plt.figure(figsize=(3, 3))
plt.imshow(random_img)
plt.axis('off')
plt.show()

score = model.evaluate(test_generator, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

score = model.evaluate(x_test, y_test)
test_loss = score[0] * 100
test_accuracy = score[1] * 100

print(f"Test Loss: {test_loss:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")

model.save("cnn_brain_tumor_model.h5")
print("Model saved as cnn_brain_tumor_model.h5")