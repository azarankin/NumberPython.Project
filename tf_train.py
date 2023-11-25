import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize, set values between 0 - 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape the input data to include the channel dimension
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

model = tf.keras.models.Sequential()

# layers
model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))  # Add the channel dimension here
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

#optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01) #prefer less

# compile
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy']) #optimizer='adam'

# Check if the folder exists and delete it if it does
model_folder = 'handwritten.model'
if os.path.exists(model_folder):
    print(f"Deleting existing folder: {model_folder}")
    os.rmdir(model_folder)

# augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20, #prefer less
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.5,
    horizontal_flip=False, #not needed
    vertical_flip=False #not needed
)

datagen.fit(x_train)

# train
model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=4)

# save the model
model.save(model_folder)
