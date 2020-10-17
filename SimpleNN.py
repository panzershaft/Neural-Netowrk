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

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def forward_propagation(self, X):
        z2 = np.dot(X, self.W1)
        s2 = self.sigmoid(z2)
        z3 = np.dot(s2, self.W2)
        yHat = self.sigmoid(z3)
        return yHat


NN = NeuralNetwork()
X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
# y = np.array(([75], [82], [93]), dtype=float)
result = NN.forward_propagation(X)
print(result)
