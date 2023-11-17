import numpy as np
import matplotlib.pyplot as plt
import load_dataset_util
import image_util
import config





weights_input_to_hidden, weights_hidden_to_output, bias_input_to_hidden, bias_hidden_to_output = load_dataset_util.set_weight(config.dataset_file, config.weights_file)



# CHECK CUSTOM
test_image = plt.imread(config.test_image, format="jpeg")


resized_image = image_util.preprocess_image(test_image)

# Predict
image = np.reshape(resized_image, (-1, 1))

# Forward propagation (to hidden layer)
hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
hidden = 1 / (1 + np.exp(-hidden_raw))  # sigmoid
# Forward propagation (to output layer)
output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
output = 1 / (1 + np.exp(-output_raw))
output_probabilities = np.exp(output) / np.sum(np.exp(output), axis=0)
max_output_probabilities = "{:.0f}".format(np.max(output_probabilities) * 100)

plt.imshow(resized_image.reshape(28, 28), cmap="Greys")
plt.title(f"NN suggests the CUSTOM number is: {output.argmax()} ({max_output_probabilities}%)")
plt.show()
