import numpy as np

# Sample dataset: words and their labels (0 = short, 1 = long)
words = ['cat', 'dog', 'elephant', 'ant', 'giraffe', 'zebra']
labels = [0, 0, 1, 0, 1, 1]  # Short words are labeled as 0, long words as 1

# Feature extraction function
def extract_features(words):
    features = []
    for word in words:
        length = len(word)
        vowels = sum(1 for char in word if char in 'aeiou')
        features.append([length, vowels])  # Length and number of vowels
    return np.array(features)

# Prepare feature matrix and label vector
X = extract_features(words)
y = np.array(labels).reshape(-1, 1)

# Neural Network Parameters
input_size = X.shape[1]  # Number of features
hidden_size = 4           # Number of neurons in hidden layer
output_size = 1           # Output layer size (binary classification)
learning_rate = 0.01
epochs = 10000

# Initialize weights and biases
np.random.seed(42)
weights_input_hidden = np.random.rand(input_size, hidden_size)
bias_hidden = np.random.rand(1, hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_output = np.random.rand(1, output_size)

# Activation function (Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Training the neural network
for epoch in range(epochs):
    # Forward Propagation
    hidden_layer_activation = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_activation)

    # Calculate error
    error = y - predicted_output

    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# Testing the neural network with new data
test_words = ['bat', 'hippopotamus', 'antelope']
test_X = extract_features(test_words)

# Forward pass for testing
hidden_layer_activation_test = np.dot(test_X, weights_input_hidden) + bias_hidden
hidden_layer_output_test = sigmoid(hidden_layer_activation_test)

output_layer_activation_test = np.dot(hidden_layer_output_test, weights_hidden_output) + bias_output
predicted_test_output = sigmoid(output_layer_activation_test)

# Classify based on threshold (0.5)
predictions = (predicted_test_output > 0.5).astype(int)

# Print results
for word, prediction in zip(test_words, predictions):
    print(f"Word: {word}, Predicted Class: {'Long' if prediction[0] == 1 else 'Short'}")