import numpy as np

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the training data (Table A)
training_data = np.array([[1, 1],
                          [1, 0],
                          [0, 1],
                          [0, 0]])

# Define the target outputs for NAND gate
target_outputs = np.array([[0],
                           [1],
                           [1],
                           [1]])

# Initialize neural network parameters
input_size = 2
hidden1_size = 2
hidden2_size = 2
output_size = 2

learning_rate = 0.2
momentum = 0.0
epochs = 10000
error_threshold = 0.1

# Initialize weights and biases with random values
np.random.seed(0)
input_to_hidden1_weights = np.random.uniform(-1, 1, (input_size, hidden1_size))
hidden1_to_hidden2_weights = np.random.uniform(-1, 1, (hidden1_size, hidden2_size))
hidden2_to_output_weights = np.random.uniform(-1, 1, (hidden2_size, output_size))
hidden1_biases = np.random.uniform(-1, 1, (1, hidden1_size))
hidden2_biases = np.random.uniform(-1, 1, (1, hidden2_size))
output_biases = np.random.uniform(-1, 1, (1, output_size))

# Training the neural network
for epoch in range(epochs):
    total_error = 0

    for i in range(len(training_data)):
        # Forward propagation
        input_layer = training_data[i:i+1]
        hidden1_input = np.dot(input_layer, input_to_hidden1_weights) + hidden1_biases
        hidden1_output = sigmoid(hidden1_input)
        hidden2_input = np.dot(hidden1_output, hidden1_to_hidden2_weights) + hidden2_biases
        hidden2_output = sigmoid(hidden2_input)
        output_layer_input = np.dot(hidden2_output, hidden2_to_output_weights) + output_biases
        output_layer_output = sigmoid(output_layer_input)

        # Calculate the error
        error = target_outputs[i:i+1] - output_layer_output
        total_error += np.sum(error**2)

        # Backpropagation
        delta_output = error * sigmoid_derivative(output_layer_output)
        error_hidden2 = delta_output.dot(hidden2_to_output_weights.T)
        delta_hidden2 = error_hidden2 * sigmoid_derivative(hidden2_output)
        error_hidden1 = delta_hidden2.dot(hidden1_to_hidden2_weights.T)
        delta_hidden1 = error_hidden1 * sigmoid_derivative(hidden1_output)

        # Update weights and biases
        hidden2_to_output_weights += learning_rate * hidden2_output.T.dot(delta_output) + momentum
        hidden1_to_hidden2_weights += learning_rate * hidden1_output.T.dot(delta_hidden2) + momentum
        input_to_hidden1_weights += learning_rate * input_layer.T.dot(delta_hidden1) + momentum
        output_biases += learning_rate * delta_output.sum(axis=0) + momentum
        hidden2_biases += learning_rate * delta_hidden2.sum(axis=0) + momentum
        hidden1_biases += learning_rate * delta_hidden1.sum(axis=0) + momentum

    if total_error < error_threshold:
        break

print(f"Training complete. Total epochs: {epoch+1}")

# Testing the neural network with modified inputs (Table B)
testing_data = np.array([[0.1, 0.2],
                         [0.1, 0.9],
                         [0.9, 0.1],
                         [0.9, 1.1]])

print("Testing Results:")
for i in range(len(testing_data)):
    input_layer = testing_data[i:i+1]
    hidden1_input = np.dot(input_layer, input_to_hidden1_weights) + hidden1_biases
    hidden1_output = sigmoid(hidden1_input)
    hidden2_input = np.dot(hidden1_output, hidden1_to_hidden2_weights) + hidden2_biases
    hidden2_output = sigmoid(hidden2_input)
    output_layer_input = np.dot(hidden2_output, hidden2_to_output_weights) + output_biases
    output_layer_output = sigmoid(output_layer_input)
    print(f"Input: {input_layer}, Output: {output_layer_output}")
