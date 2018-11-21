import numpy as np
import utils

from neural_network import NeuralNetwork

if __name__ == '__main__':
    X = np.concatenate((np.random.normal(0., 1., (3, 2)),
                        np.random.normal(2., 1., (2, 2))),
                       axis=0)
    y = np.array([1, 1, 1, -1, -1])

    topology = [X.shape[1], 3, 2]
    nn = NeuralNetwork(X, y, topology, 'sigmoid', 1000)

    nn.train(.05)

    utils.plot_learning_curve(nn.empirical_risk, nn.max_epochs)
