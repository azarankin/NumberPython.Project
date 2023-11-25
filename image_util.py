
import cv2
import numpy as np

def preprocess_image(image_data):

    if len(image_data.shape) == 2:  #returned image upload
        image_data = 1 - np.expand_dims(image_data, axis=-1).astype("float32") / 255
        
    else:
        gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        image_data = 1 - (gray(image_data).astype("float32") / 255)

    # Resize the image to 28x28 using OpenCV
    image_data = cv2.resize(image_data, (28, 28))

        # Ensure the image has a single channel (grayscale)
    if len(image_data.shape) == 3:
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    
    #image_data = np.invert(image_data)
    image_data = image_data.reshape(1, 28, 28)

    return image_data