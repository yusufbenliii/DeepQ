# combination of exponentiation and normalization is softmax

import numpy as np
import nnfs
from nnfs.datasets import spiral_data


class Layer_Dense:  # input to layer
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:  # layer to layer
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:  # layer to output
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


nnfs.init()
X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

# layer_outputs = [[4.8, 1.21, 2.385],
#                  [8.9, -1.81, 0.2],
#                  [1.41, 1.051, 0.026]]
# exp_values = np.exp(layer_outputs)
# print(exp_values)
# print(np.sum(layer_outputs, axis=1, keepdims=True))
# norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
# print(norm_values)
# print(np.sum(norm_values, axis=1, keepdims=True))


# #E = 2.71828182846
# E = np.e
# layer_outputs = [4.8, 1.21, 2.385]
# exp_values = []
# for output in layer_outputs:
#     exp_values.append(E**output)
# print(exp_values)
#
#
# norm_base = np.sum(exp_values)
# norm_values = []
# for value in exp_values:
#     norm_values.append(value/norm_base)
# print(norm_values)
# print(np.sum(norm_values))
