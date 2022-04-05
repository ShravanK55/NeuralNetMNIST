"""
Neural Network - Multi-Layer Perceptron

A module an MLP neural network to classify digits from the MNIST dataset.

Author: shravan@usc.edu (5451873903)

"""

import numpy as np
import sys


def sigmoid(x):
    """
    Sigmoid activation function.

    Args:
        x(np.array): Matrix to perform the sigmoid activation for.

    Returns:
        (np.array): Matrix computed using the sigmoid activation function on the input.

    """
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    """
    Softmax activation function.

    Args:
        x(np.array): Matrix to perform the softmax activation for.

    Returns:
        (np.array): Matrix computed using the softmax activation function on the input.

    """
    inp_exp = np.exp(x)
    return inp_exp / np.sum(inp_exp, axis=1, keepdims=True)


class NeuralNetwork:
    """
    Module implementing a multi-layer perceptron neural network.
    """

    def __init__(self, model, learning_rate=0.01, batch_size=64, epochs=50):
        """
        Method to initialize the neural network.

        Args:
            model(tuple): Structure of the neural network. Must have 4 elements. Format: (
                    input_layer_nodes(int),
                    hidden_layer_1_nodes(int),
                    hidden_layer_2_nodes(int),
                    output_layer_nodes(int)
                )
            learning_rate(float): Learning rate of the neural network. Defaults to 0.01.
            batch_size(int): Batch size of inputs to use for training. Defaults to 64.
            epochs(int): Number of iterations to run the training for. Defaults to 50.

        """
        # Hyper parameters for training.
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # Weights of the connections between each layer.
        self.w1 = np.random.rand(model[0], model[1]) * np.sqrt(1.0 / model[0])
        self.w2 = np.random.rand(model[1], model[2]) * np.sqrt(1.0 / model[1])
        self.w_out = np.random.rand(model[2], model[3]) * np.sqrt(1.0 / model[2])

        # Biases for activation at each layer.
        self.b1 = np.random.rand(1, model[1]) * np.sqrt(1.0 / model[0])
        self.b2 = np.random.rand(1, model[2]) * np.sqrt(1.0 / model[1])
        self.b_out = np.random.rand(1, model[3]) * np.sqrt(1.0 / model[2])

        # Cache of the weighted sum at each layer.
        self.ws1 = []
        self.ws2 = []
        self.ws_out = []

        # Output at each layer from the activation function.
        self.out1 = []
        self.out2 = []
        self.out_final = []

    def forward_pass(self, input):
        """
        Method to perform a forward pass through the neural network.

        Args:
            input(np.array): Matrix containing an input batch.

        """
        self.ws1 = np.dot(input, self.w1) + self.b1
        self.out1 = sigmoid(self.ws1)

        self.ws2 = np.dot(self.ws1, self.w2) + self.b2
        self.out2 = sigmoid(self.ws2)

        self.ws_out = np.dot(self.ws2, self.w_out) + self.b_out
        self.out_final = softmax(self.ws_out)

    def classify(self, input):
        """
        Method to perform classification of the input from the neural network.

        Args:
            input(np.array): Matrix containing an input batch.

        Returns:
            (np.array): Array with the predictions for the input batch.

        """
        self.forward_pass(input)
        return np.argmax(self.out_final, axis=1)


if __name__ == "__main__":
    train_img_path, train_label_path, test_img_path = sys.argv[1], sys.argv[2], sys.argv[3]
    TEST_LABEL_PATH = "data/test_label.csv"
    OUTPUT_PATH = "data/test_predictions.csv"

    train_images = np.loadtxt(train_img_path, dtype=np.float64, delimiter=',')
    train_labels = np.loadtxt(train_label_path, dtype=int, delimiter=',')
    test_images = np.loadtxt(test_img_path, dtype=np.float64, delimiter=',')
    test_labels = np.loadtxt(TEST_LABEL_PATH, dtype=int, delimiter=',')

    # Convert the values in the dataset to the range of 0-1.
    train_images /= 255
    test_images /= 255

    neural_network = NeuralNetwork((784, 512, 256, 10))
    predictions = neural_network.classify(train_images[:32])
