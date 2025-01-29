"""
Building a CNN Model to Classify Images from Popular Datasets (MNIST, CIFAR-10, ImageNet)
"""

import tensorflow as tf #type:ignore
from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout #type:ignore
from tensorflow.keras.datasets import cifar10 #type:ignore
from tensorflow.keras.utils import to_categorical #type:ignore

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),  # Convolutional layer
    MaxPooling2D((2, 2)),  # Max pooling layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),  # Flatten feature maps to a vector
    Dense(256, activation='relu'),  # Fully connected layer
    Dropout(0.5),  # Dropout for regularization
    Dense(10, activation='softmax')  # Output layer for 10 classes (CIFAR-10 categories)
])

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
history = model.fit(
    x_train, y_train, 
    epochs=5, 
    batch_size=64, 
    validation_split=0.2, 
    verbose=1
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")