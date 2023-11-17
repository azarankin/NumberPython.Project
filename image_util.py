
import cv2
import numpy as np

def preprocess_image(image_data):
    # Grayscale + Unit RGB + inverse colors
    gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    test_image = 1 - (gray(image_data).astype("float32") / 255)

    # Resize the image to 28x28 using OpenCV
    resized_image = cv2.resize(test_image, (28, 28))

    # Reshape
    resized_image = np.reshape(resized_image, (resized_image.shape[0] * resized_image.shape[1]))

    return resized_image