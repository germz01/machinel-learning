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


nn = neural_network.NeuralNetwork(hidden_sizes=(3,20,7 ),
                                  activation='sigmoid', max_epochs=500)

nn.train(X, y, eta=0.5, alpha=0)
nn.y_pred

for iy in nn.delta:
        print iy.shape

for delta_W in nn.delta_W:
        print delta_W.shape

for W in nn.W:
        print W.shape


if raw_input('\nDO YOU WANT TO PLOT THE LEARNING CURVE?[Y/N] ') == 'Y':
        utils.plot_learning_curve(nn.empirical_risk, nn.max_epochs)


