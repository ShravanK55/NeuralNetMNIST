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
        x(np.array): Matrix to apply the sigmoid activation for.

    Returns:
        (np.array): Matrix computed using the sigmoid activation function on the input.

    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Derivative of the Sigmoid activation function.

    Args:
        x(np.array): Matrix to apply the sigmoid activation derivative for.

    Returns:
        (np.array): Matrix computed using the sigmoid activation function derivative on the input.

    """
    return x * (1 - x)


def softmax(x):
    """
    Softmax activation function.

    Args:
        x(np.array): Matrix to perform the softmax activation for.

    Returns:
        (np.array): Matrix computed using the softmax activation function on the input.

    """
    inp_exp = np.exp(x)
    return inp_exp / np.sum(inp_exp, axis=0, keepdims=True)


def classify(out):
    """
    Method to get the classification for images from the output of the neural network.

    Args:
        out(np.array): Output of the neural network.

    Returns:
        (np.array): Classification of the images as a single dimensional array.

    """
    return np.argmax(out, axis=0)


class NeuralNetwork:
    """
    Module implementing a multi-layer perceptron neural network.
    """

    def __init__(self, model, learning_rate=0.01, batch_size=32, epochs=30):
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
            batch_size(int): Batch size of inputs to use for training. Defaults to 32.
            epochs(int): Number of iterations to run the training for. Defaults to 30.

        """
        # Hyper parameters for training.
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # Initialization of weights and biases using Xavier initialization.
        # Weights of the connections between each layer.
        self.w1 = np.random.randn(model[1], model[0]) * np.sqrt(1.0 / (model[0] + model[1]))
        self.w2 = np.random.randn(model[2], model[1]) * np.sqrt(1.0 / (model[1] + model[2]))
        self.w_out = np.random.randn(model[3], model[2]) * np.sqrt(1.0 / (model[2] + model[3]))

        # Biases for activation at each layer.
        self.b1 = np.random.randn(model[1], 1) * np.sqrt(1.0 / (model[0] + model[1]))
        self.b2 = np.random.randn(model[2], 1) * np.sqrt(1.0 / (model[1] + model[2]))
        self.b_out = np.random.randn(model[3], 1) * np.sqrt(1.0 / (model[2] + model[3]))

        # Cache of the weighted sum at each layer.
        self.ws1 = []
        self.ws2 = []
        self.ws_out = []

        # Output at each layer from the activation function.
        self.out1 = []
        self.out2 = []
        self.out_final = []

    def forward_pass(self, input_batch):
        """
        Method to perform a forward pass through the neural network.

        Args:
            input_batch(np.array): Matrix containing an input batch.

        Returns:
            (np.array): Matrix with the output of the neural network.

        """
        self.ws1 = np.dot(self.w1, input_batch) + self.b1
        self.out1 = sigmoid(self.ws1)

        self.ws2 = np.dot(self.w2, self.ws1) + self.b2
        self.out2 = sigmoid(self.ws2)

        self.ws_out = np.dot(self.w_out, self.ws2) + self.b_out
        self.out_final = softmax(self.ws_out)
        return self.out_final

    def backward_pass(self, input_batch, expected_output):
        """
        Method to perform a backward pass to update weights and biases in the neural network.

        Args:
            input_batch(np.array): Matrix containing an input batch.
            expected_output(np.array): Expected output of the neural network with one hot encoding.

        """
        # Error at the output layer.
        d_out_final = self.out_final - expected_output

        # Backpropagation parameters for weights and biases at the output layer.
        d_w_out = np.dot(d_out_final, self.out2.T) / self.batch_size
        d_b_out = np.sum(d_out_final, axis=1, keepdims=True) / self.batch_size

        # Backpropagation parameters for weights and biases at the second hidden layer.
        d_out2 = np.dot(self.w_out.T, d_out_final)
        d_ws2 = d_out2 * sigmoid_derivative(self.out2)
        d_w2 = np.dot(d_ws2, self.out1.T) / self.batch_size
        d_b2 = np.sum(d_ws2, axis=1, keepdims=True) / self.batch_size

        # Backpropagation parameters for weights and biases at the first hidden layer.
        d_out1 = np.dot(self.w2.T, d_ws2)
        d_ws1 = d_out1 * sigmoid_derivative(self.out1)
        d_w1 = np.dot(d_ws1, input_batch.T) / self.batch_size
        d_b1 = np.sum(d_ws1, axis=1, keepdims=True) / self.batch_size

        # Updating weights and biases.
        self.w1 = self.w1 - (self.learning_rate * d_w1)
        self.b1 = self.b1 - (self.learning_rate * d_b1)
        self.w2 = self.w2 - (self.learning_rate * d_w2)
        self.b2 = self.b2 - (self.learning_rate * d_b2)
        self.w_out = self.w_out - (self.learning_rate * d_w_out)
        self.b_out = self.b_out - (self.learning_rate * d_b_out)

    def train(self, images, labels):
        """
        Method to train the neural network.

        Args:
            images(np.array): Training image data. Must be of dimension (num_samples, image_size).
            labels(np.array): Training image labels with one hot encoding. Must be of dimension
                (num_samples, num_classifcations).

        """
        for epoch in range(self.epochs):
            print("Epoch {}.".format(epoch))
            num_samples = len(images)
            num_correct = 0
            for batch_idx in range(0, len(images), self.batch_size):
                self.forward_pass(images[batch_idx : batch_idx + self.batch_size].T)
                self.backward_pass(images[batch_idx : batch_idx + self.batch_size].T,
                                   labels[batch_idx : batch_idx + self.batch_size].T)
                output = classify(self.out_final)
                expected_output = classify(labels[batch_idx : batch_idx + self.batch_size].T)
                num_correct += len([0 for i in range(len(output)) if output[i] == expected_output[i]])

            accuracy = num_correct / num_samples
            print("Accuracy: {}".format(accuracy))


if __name__ == "__main__":
    train_img_path, train_label_path, test_img_path = sys.argv[1], sys.argv[2], sys.argv[3]
    TEST_LABEL_PATH = "data/test_label.csv"
    OUTPUT_PATH = "data/test_predictions.csv"

    train_images = np.loadtxt(train_img_path, dtype=np.float64, delimiter=',') / 255
    train_labels = np.loadtxt(train_label_path, dtype=int, delimiter=',')
    test_images = np.loadtxt(test_img_path, dtype=np.float64, delimiter=',') / 255
    test_labels = np.loadtxt(TEST_LABEL_PATH, dtype=int, delimiter=',')

    # Convert labels to one hot encoding for easier backpropagation.
    one_hot_train_labels = np.zeros((train_labels.size, train_labels.max() + 1))
    one_hot_train_labels[np.arange(train_labels.size), train_labels] = 1
    one_hot_test_labels = np.zeros((test_labels.size, test_labels.max() + 1))
    one_hot_test_labels[np.arange(test_labels.size), test_labels] = 1

    # Creating and training the neural network.
    neural_network = NeuralNetwork((784, 512, 256, 10))
    neural_network.train(train_images, one_hot_train_labels)

    # Getting the predictions from the testing dataset and getting the prediction accuracy.
    nn_output = neural_network.forward_pass(test_images)
    output = classify(nn_output)
    expected_output = classify(one_hot_test_labels.T)
    num_correct = len([0 for i in range(len(output)) if output[i] == expected_output[i]])
    num_samples = len(test_images)
    print("Testing dataset accuracy: {}".format(num_correct / num_samples))
