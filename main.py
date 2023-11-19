from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from io import BytesIO
import uuid
import load_dataset_util
import image_util
import config


weights_input_to_hidden, weights_hidden_to_output, bias_input_to_hidden, bias_hidden_to_output = load_dataset_util.set_weight(config.dataset_file, config.weights_file)

app = Flask(__name__)

files_dir = 'files'
if not os.path.exists(files_dir):
    os.makedirs(files_dir)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/files/<path:filename>')
def serve_file(filename):
    return send_from_directory(files_dir, filename)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 401

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 402

    if file:
        image_data = plt.imread(BytesIO(file.read()), format="jpeg")

        # Preprocess the image
        processed_image = image_util.preprocess_image(image_data)

        # Predict
        image = np.reshape(processed_image, (-1, 1))

        # Forward propagation (to hidden layer)
        hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
        hidden = 1 / (1 + np.exp(-hidden_raw))  # sigmoid
        # Forward propagation (to output layer)
        output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
        output = 1 / (1 + np.exp(-output_raw))
        output_probabilities = np.exp(output) / np.sum(np.exp(output), axis=0)
        max_output_probabilities = "{:.0f}".format(np.max(output_probabilities) * 100)

        predicted_number = output.argmax()
        probability = int(max_output_probabilities)

        # Invert the colors of the processed image
        inverted_image = 1 - processed_image

        # Save the inverted image using OpenCV
        unique_filename = str(uuid.uuid4())
        file_path = os.path.join('files', f'{unique_filename}.jpg')

        # Ensure processed image has the correct shape (28x28)

        inverted_image = cv2.resize(inverted_image.reshape(28, 28), (28, 28))

        # Save the inverted image
        cv2.imwrite(file_path, (inverted_image * 255).astype(np.uint8))  # Convert to uint8 before saving
        file_path = file_path.replace('\\', '/')

        # Return the result as JSON
        result = {'number': int(predicted_number), 'probability': probability, 'image_address': file_path}
        return jsonify(result), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.getenv("PORT", default=5000))
