import numpy as np
import utils
import neural_network
import imp

X = np.concatenate((np.random.normal(0., 1., (3, 2)),
                    np.random.normal(3., 1., (2, 2))),
                   axis=0)
y = np.array([1, 1, 1, -1, -1])

# reloading del module per sviluppo
imp.reload(neural_network)
imp.reload(utils)


nn = neural_network.NeuralNetwork(hidden_sizes=(100, 3, 2),
                                  activation='sigmoid', max_epochs=500)
nn.train(X, y, eta=2, alpha=0.7)
nn.y_pred


if raw_input('\nDO YOU WANT TO PLOT THE LEARNING CURVE?[Y/N] ') == 'Y':
        utils.plot_learning_curve(nn.empirical_risk, nn.max_epochs)

# ann prediction

