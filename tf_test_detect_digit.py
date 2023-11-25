import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import image_util
from io import BytesIO

model = tf.keras.models.load_model('handwritten.model')

# 3
img1 = plt.imread('test1.jpg')  # Read the image in grayscale
img1 = image_util.preprocess_image(img1)
prediction1 = model.predict(img1)
print(f'For 3, the prediction is: {np.argmax(prediction1)}')
plt.imshow(img1[0], cmap=plt.cm.binary)  # Display the reshaped image
plt.show()


# 2
img2 = plt.imread('test2.jpg')
img2 = image_util.preprocess_image(img2)
prediction2 = model.predict(img2)
print(f'For 2, the prediction is: {np.argmax(prediction2)}')
plt.imshow(img2[0], cmap=plt.cm.binary)  # Display the reshaped image
plt.show()


# try to train with misclassification value