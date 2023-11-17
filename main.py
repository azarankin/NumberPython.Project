import numpy as np
import matplotlib.pyplot as plt
import os

import load_dataset


dataset_file = "mnist.npz"
weights_file = 'weights.npz'


weights_input_to_hidden, weights_hidden_to_output, bias_input_to_hidden, bias_hidden_to_output = load_dataset.set_weight(dataset_file, weights_file)

# CHECK CUSTOM
test_image = plt.imread("custom.jpg", format="jpeg")

# Grayscale + Unit RGB + inverse colors
gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
test_image = 1 - (gray(test_image).astype("float32") / 255)

# Reshape
test_image = np.reshape(test_image, (test_image.shape[0] * test_image.shape[1]))

# Predict
image = np.reshape(test_image, (-1, 1))

# Forward propagation (to hidden layer)
hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
hidden = 1 / (1 + np.exp(-hidden_raw))  # sigmoid
# Forward propagation (to output layer)
output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
output = 1 / (1 + np.exp(-output_raw))

plt.imshow(test_image.reshape(28, 28), cmap="Greys")
plt.title(f"NN suggests the CUSTOM number is: {output.argmax()}")
plt.show()
