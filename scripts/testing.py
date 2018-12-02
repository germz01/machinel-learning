import numpy as np
import utils
import matplotlib.pyplot as plt
import neural_network
import imp

X = np.concatenate((np.random.normal(2., 1., (3, 2)),
                    np.random.normal(5., 1., (2, 2))),
                   axis=0)

# Y = X[]
# y = np.round((2*X[:, 0]+5*X[:, 1]+10*X[:,0]*X[:,1])/20)
y = np.array([1, 1, 1, -1, -1])


# reloading del module per sviluppo
imp.reload(neural_network)
imp.reload(utils)

momentum = raw_input('MOMENTUM[classic/nesterov]: ')

nn = neural_network.NeuralNetwork(hidden_sizes=(20, 5),
                                  activation='sigmoid', max_epochs=1000,
                                  max_weight_init=0.7)

nn.train(X, y, eta=0.2, alpha=0.7)

np.round(nn.y_pred, 2)

utils.plot_learning_curve([nn.empirical_risk, nn.error_rmse, nn.error_mee,
                          np.array(nn.error_mee)+np.array(nn.error_mee_dev)],
                          nn.max_epochs, momentum=momentum,
                          fname=['empirical_risk_', 'error_rmse_',
                          'error_mee_', 'error_mee_dev_'])


# plt.plot(range(nn.max_epochs), np.array(nn.error_mee)+np.
#          array(nn.error_mee_dev))
# plt.plot(range(nn.max_epochs), np.array(nn.error_mee)-np.
#          array(nn.error_mee_dev))
# plt.plot(range(nn.max_epochs), np.array(nn.error_mee))
# plt.show()

# plt.plot(range(nn.max_epochs), np.array(nn.error_mee_dev))
# plt.plot(range(nn.max_epochs), -np.array(nn.error_mee_dev))
# plt.show()


for iy in nn.Y:
        print iy.shape

for d in nn.delta:
        print d.shape

for delta_W in nn.delta_W:
        print delta_W.shape

for W in nn.W:
        print W.shape


# if raw_input('\nDO YOU WANT TO PLOT THE LEARNING CURVE?[Y/N] ') == 'Y':
#        utils.plot_learning_curve(nn.empirical_risk, nn.max_epochs)


###########################################################

# testing error functions

p = 4
k = 1

# d = np.ones((p,k)).reshape((p,))

d = np.arange(p*k).reshape((p, 1))
y = np.ones((p, k))

d
y

# rmse

print np.sqrt(np.einsum('pk->', (d-y)**2) / p)

# mean euclidean error
print np.einsum('p->', np.sqrt(np.einsum('pk->p', (d-y)**2))) / p


def rmse(d, y):
        '''root mean square error'''
        p = d.shape[0]
        if len(d.shape) == 1:
                d = d.reshape((p, 1))
        if len(y.shape) == 1:
               y = y.reshape((p, 1))

        return np.sqrt(np.einsum('pk->', (d-y)**2) / p)


def mee(d, y):
        ''' mean euclidean error'''
        p = d.shape[0]
        if len(d.shape) == 1:
                d = d.reshape((p, 1))
        if len(y.shape) == 1:
               y = y.reshape((p, 1))

        return np.einsum('p->', np.sqrt(np.einsum('pk->p', (d-y)**2))) / p


# y = np.round(np.random.uniform(-1,1, (p,k)),1)

