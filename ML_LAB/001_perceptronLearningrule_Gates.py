''' Implement the perceptron learning algorithm and apply it to solve the following problems for N-input gates and compare the output in terms of Accuracy.

AND gate
OR gate
NAND gate
XOR gate
Note: Take user input(At least Two) for gates'''

#FOR 2 INPUTS--

import numpy as np

# Define the perceptron learning algorithm
def perceptron_learning_algorithm(X, y, learning_rate=0.1, num_epochs=1000):
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)
    bias = 0

    for _ in range(num_epochs):
        for i in range(num_samples):
            y_pred = np.dot(X[i], weights) + bias
            if y_pred > 0:
                y_pred = 1
            else:
                y_pred = 0
            weights += learning_rate * (y[i] - y_pred) * X[i]
            bias += learning_rate * (y[i] - y_pred)

    return weights, bias

# Define logic gates
def and_gate(X):
    weights, bias = perceptron_learning_algorithm(X, [1, 1, 1, 0])
    return weights, bias

def or_gate(X):
    weights, bias = perceptron_learning_algorithm(X, [1, 1, 1, 0])
    return weights, bias

def nand_gate(X):
    weights, bias = perceptron_learning_algorithm(X, [0, 0, 0, 1])
    return weights, bias

def xor_gate(X):
    weights1, bias1 = perceptron_learning_algorithm(X, [0, 1, 1, 0])
    weights2, bias2 = perceptron_learning_algorithm(X, [0, 1, 1, 0])

    return (weights1, bias1), (weights2, bias2)

# Take user input for the gate
gate = input("Enter the gate (AND, OR, NAND, XOR): ").upper()

# Define the input data for all gates
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

if gate == "AND":
    weights, bias = and_gate(X)
elif gate == "OR":
    weights, bias = or_gate(X)
elif gate == "NAND":
    weights, bias = nand_gate(X)
elif gate == "XOR":
    weights1, bias1, weights2, bias2 = xor_gate(X)
else:
    print("Invalid gate name")
    exit()

# Test the gate and calculate accuracy
correct = 0
total = len(X)
if gate == "XOR":
    for x in X:
        y1_pred = np.dot(x, weights1) + bias1
        y2_pred = np.dot(x, weights2) + bias2
        if (y1_pred > 0 and y2_pred <= 0) or (y1_pred <= 0 and y2_pred > 0):
            correct += 1
else:
    for x in X:
        y_pred = np.dot(x, weights) + bias
        if (y_pred > 0 and gate == "AND") or (y_pred > 0 and gate == "OR") or (y_pred <= 0 and gate == "NAND"):
            correct += 1

accuracy = (correct / total) * 100
print(f"Accuracy of the {gate} gate: {accuracy}%")




#FOR 3 INPUTS---------------

import numpy as np

# Define the perceptron learning algorithm
def perceptron_learning_algorithm(X, y, learning_rate=0.1, num_epochs=1000):
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)
    bias = 0

    for _ in range(num_epochs):
        for i in range(num_samples):
            y_pred = np.dot(X[i], weights) + bias

            # Update weights and bias only when predicted output is not equal to target output
            if y_pred > 0:
                y_pred = 1
            else:
                y_pred = 0
            
            if y_pred != y[i]:
                weights += learning_rate * (y[i] - y_pred) * X[i]
                bias += learning_rate * (y[i] - y_pred)

    return weights, bias

# Define logic gates for 2 inputs
def and_gate_2_inputs(X):
    weights, bias = perceptron_learning_algorithm(X, [1, 0, 0, 0])
    return weights, bias

def or_gate_2_inputs(X):
    weights, bias = perceptron_learning_algorithm(X, [0, 1, 1, 1])
    return weights, bias

def nand_gate_2_inputs(X):
    weights, bias = perceptron_learning_algorithm(X, [0, 1, 1, 1])
    return weights, bias

def xor_gate_2_inputs(X):
    weights1, bias1 = perceptron_learning_algorithm(X, [0, 1, 1, 0])
    weights2, bias2 = perceptron_learning_algorithm(X, [0, 0, 0, 1])

    return (weights1, bias1), (weights2, bias2)

# Define logic gates for 3 inputs
def and_gate_3_inputs(X):
    weights, bias = perceptron_learning_algorithm(X, [1, 0, 0, 0, 0, 0, 0, 0])
    return weights, bias

def or_gate_3_inputs(X):
    weights, bias = perceptron_learning_algorithm(X, [0, 1, 1, 1, 1, 1, 1, 1])
    return weights, bias

def nand_gate_3_inputs(X):
    weights, bias = perceptron_learning_algorithm(X, [0, 1, 1, 1, 1, 1, 1, 1])
    return weights, bias

def xor_gate_3_inputs(X):
    weights, bias = perceptron_learning_algorithm(X, [0, 1, 1, 0, 1, 0, 0, 1])
    return weights, bias

# Take user input for the gate and number of inputs
gate = input("Enter the gate (AND, OR, NAND, XOR): ").upper()
n_inputs = int(input("Enter the number of inputs (2 or 3): "))

# Define the input data for the selected number of inputs
if n_inputs == 2:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
else:
    X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

# Depending on the selected gate and number of inputs, call the respective function to train a perceptron (or perceptrons)
if n_inputs == 2:
    if gate == "AND":
        weights, bias = and_gate_2_inputs(X)
    elif gate == "OR":
        weights, bias = or_gate_2_inputs(X)
    elif gate == "NAND":
        weights, bias = nand_gate_2_inputs(X)
    elif gate == "XOR":
        weights1, bias1, weights2, bias2 = xor_gate_2_inputs(X)
    else:
        print("Invalid gate name")
        exit()
else:
    if gate == "AND":
        weights, bias = and_gate_3_inputs(X)
    elif gate == "OR":
        weights, bias = or_gate_3_inputs(X)
    elif gate == "NAND":
        weights, bias = nand_gate_3_inputs(X)
    elif gate == "XOR":
        weights, bias = xor_gate_3_inputs(X)
    else:
        print("Invalid gate name")
        exit()

# Test the gate's predictions and calculate accuracy
correct = 0
total = len(X)

if gate == "XOR":
    for x in X:
        y_pred = np.dot(x, weights) + bias
        if (y_pred > 0 and x[0] != 1) or (y_pred <= 0 and x[0] == 1):
            correct += 1
else:
    for x in X:
        y_pred = np.dot(x, weights) + bias
        if (y_pred > 0 and gate == "AND") or (y_pred > 0 and gate == "OR") or (y_pred <= 0 and gate == "NAND"):
            correct += 1

accuracy = (correct / total) * 100
print(f"Accuracy of the {gate} gate with {n_inputs} inputs: {accuracy}%")
