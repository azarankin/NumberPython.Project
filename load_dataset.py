import os
import numpy as np

def load_dataset(data_set_file):
	assert os.path.exists(data_set_file)
	with np.load(data_set_file) as f:
		# convert from RGB to Unit RGB
		x_train = f['x_train'].astype("float32") / 255

		# reshape from (60000, 28, 28) into (60000, 784)
		x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

		# labels
		y_train = f['y_train']

		# convert to output layer format
		y_train = np.eye(10)[y_train] # 64 * 10

		return x_train, y_train
	



# Check if the weights file exists



def set_weight(data_set_file, weights_file):
    if os.path.exists(weights_file):
        # Load weights from file
        with np.load(weights_file) as data:
            weights_input_to_hidden = data['weights_input_to_hidden']
            weights_hidden_to_output = data['weights_hidden_to_output']
            bias_input_to_hidden = data['bias_input_to_hidden']
            bias_hidden_to_output = data['bias_hidden_to_output']
    else:
        # Initialize weights if the file doesn't exist
        weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
        weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20))
        bias_input_to_hidden = np.zeros((20, 1))
        bias_hidden_to_output = np.zeros((10, 1))

        images, labels = load_dataset(data_set_file)

        epochs = 3
        e_loss = 0
        e_correct = 0
        learning_rate = 0.01

        for epoch in range(epochs):
            print(f"Epoch №{epoch}")

            for image, label in zip(images, labels):
                image = np.reshape(image, (-1, 1))
                label = np.reshape(label, (-1, 1))

                # Forward propagation (to hidden layer)
                hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
                hidden = 1 / (1 + np.exp(-hidden_raw))  # sigmoid

                # Forward propagation (to output layer)
                output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
                output = 1 / (1 + np.exp(-output_raw))

                # Loss / Error calculation
                e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
                e_correct += int(np.argmax(output) == np.argmax(label))

                # Backpropagation (output layer)
                delta_output = output - label
                weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
                bias_hidden_to_output += -learning_rate * delta_output

                # Backpropagation (hidden layer)
                delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
                weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
                bias_input_to_hidden += -learning_rate * delta_hidden

            # Save weights after each epoch
            np.savez(weights_file,
                    weights_input_to_hidden=weights_input_to_hidden,
                    weights_hidden_to_output=weights_hidden_to_output,
                    bias_input_to_hidden=bias_input_to_hidden,
                    bias_hidden_to_output=bias_hidden_to_output)

            # Print some debug info between epochs
            print(f"Loss: {round((e_loss[0] / images.shape[0]) * 100, 3)}%")
            print(f"Accuracy: {round((e_correct / images.shape[0]) * 100, 3)}%")
            e_loss = 0
            e_correct = 0
    return weights_input_to_hidden, weights_hidden_to_output, bias_input_to_hidden, bias_hidden_to_output
