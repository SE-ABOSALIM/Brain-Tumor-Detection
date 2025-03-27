import requests
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("C:/Users/90538/Documents/GitHub/Brain-Tumor-Detection/cnn_brain_tumor_model.h5")

print("Modelin Beklediği Giriş Şekli:", model.input_shape)

_, image_size, _, channels = model.input_shape

labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

def process_image(image):
    if isinstance(image, np.ndarray):
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Beklenmeyen giriş formatı!")

    img = cv2.bilateralFilter(img, 2, 50, 50)
    img = cv2.resize(img, (image_size, image_size))

    if channels == 3:
        img = np.stack((img,)*3, axis=-1)

    return img

img_url = 'https://raw.githubusercontent.com/SE-ABOSALIM/Brain-Tumor-Detection/refs/heads/main/cleaned-Dataset/Testing/notumor/Te-no_0035.jpg'
response = requests.get(img_url, stream=True)

img = Image.open(response.raw)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

img = np.asarray(img)
img = process_image(img)

img = img.reshape(1, image_size, image_size, channels)
img = img.astype(np.float32) / 255.0

prediction = model.predict(img)
predicted_class = np.argmax(prediction, axis=1)[0]
predicted_label = labels[predicted_class]
confidence = prediction[0][predicted_class]

print(f"Modelin Tahmini: {predicted_label} (Sınıf: {predicted_class})")
print(f"Güven Oranı: {confidence*100:.2f}%")