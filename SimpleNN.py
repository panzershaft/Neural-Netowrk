import numpy as np


class NeuralNetwork:
    """
    Base code for understanding how NN work
    """

    def __init__(self):
        """
        These parameters are known as Hyper-parameters
        They need to be decided before we start training, there
        is no formula which tells how many hidden layers/input layers are
        to be used. I think its heuristic no rule of thumb.
        """
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        self.z2 = 0
        self.a2 = 0
        self.z3 = 0

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        # Gradient of sigmoid
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def forward_propagation(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def cost_function(self, X, y):
        yHat = self.forward_propagation(X)
        J = 0.5 * sum((y - yHat) ** 2)
        return J

    def optimizer(self, X, y):
        yHat = self.forward_propagation(X)

        delta3 = np.multiply(-(y - yHat), self.sigmoid_prime(self.z3))

        dJdW2 = np.multiply(self.a2, delta3)

        delta2 = np.multiply(delta3, self.W2.T) * self.sigmoid_prime(self.z3)

        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2


NN = NeuralNetwork()
X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)
X = X / np.amax(X, axis=0)
y = y / 100  # Max test score is 100
cost1 = NN.cost_function(X, y)

dJdW1, dJdW2 = NN.optimizer(X, y)

scalar = 3
NN.W1 = NN.W1 + scalar * dJdW1
NN.W2 = NN.W2 + scalar * dJdW2
cost2 = NN.cost_function(X, y)
print(cost1, cost2)
#
dJdW1, dJdW2 = NN.optimizer(X, y)
NN.W1 = NN.W1 - scalar * dJdW1
NN.W2 = NN.W2 - scalar * dJdW2
cost3 = NN.cost_function(X, y)
print(cost2, cost3)
