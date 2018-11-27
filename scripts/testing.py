import numpy as np
import utils
import neural_network
import imp

X = np.concatenate((np.random.normal(2., 1., (3, 2)),
                    np.random.normal(5., 1., (2, 2))),
                   axis=0)

#Y = X[]
#y = np.round((2*X[:, 0]+5*X[:, 1]+10*X[:,0]*X[:,1])/20)
y = np.array([1, 1, 1, -1, -1])



# reloading del module per sviluppo
imp.reload(neural_network)
imp.reload(utils)


nn = neural_network.NeuralNetwork(hidden_sizes=(10,5,2),
                                  activation='sigmoid', max_epochs=1000,
                                  max_weight_init = 0.7)

nn.train(X, y, eta= 0.1 , alpha= 0.7)
np.round(nn.y_pred,2)
#utils.plot_learning_curve(nn.empirical_risk, nn.max_epochs)


#import matplotlib.pyplot as plt

for iy in nn.Y:
        print iy.shape

for d in nn.delta:
        print d.shape

for delta_W in nn.delta_W:
        print delta_W.shape

for W in nn.W:
        print W.shape


if raw_input('\nDO YOU WANT TO PLOT THE LEARNING CURVE?[Y/N] ') == 'Y':
        utils.plot_learning_curve(nn.empirical_risk, nn.max_epochs)
