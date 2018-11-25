from __future__ import division

import numpy as np
from utils import activation_function
from utils import add_bias_mul
from utils import compose_topology


class NeuralNetwork(object):
    """Simple implementation of an Artificial Neural Network"""

    def __init__(self, hidden_sizes, activation='sigmoid',
                 max_epochs=1000, max_weight_init=0.7):
        self.hidden_sizes = hidden_sizes
        self.n_layers = len(hidden_sizes) + 1

        self.activation_function = activation
        self.V = [0 for i in range(self.n_layers)]
        self.Y = [0 for i in range(self.n_layers)]
        self.delta = [0 for i in range(self.n_layers)]
        self.delta_W = [0 for i in range(self.n_layers)]
        self.max_epochs = max_epochs
        self.max_weight_init = max_weight_init

    def init_weights(self):
        """ """
        self.W = list()

        for i in range(1, len(self.topology)):
            self.W.append(np.random.uniform(
                -self.max_weight_init, self.max_weight_init,
                (self.topology[i], self.topology[i - 1] + 1)))

    def init_weights_test(self):
        """ """
        'weights init per testing'
        self.W = list()

        for i in range(1, len(self.topology)):
            self.W.append(np.ones((self.topology[i], self.topology[i - 1]+1)))

    def target_scale(self, y):
        """

        Parameters
        ----------
        y :


        Returns
        -------

        """
        if self.activation_function == 'sigmoid':
            MIN = y.min()
            MAX = y.max()
            return (y-MIN) / (MAX-MIN)
        else:
            # da implementare scaling per altre activations
            return NotImplemented

    def target_scale_back(self, y_pred):
        """

        Parameters
        ----------
        y_pred :


        Returns
        -------

        """
        if self.activation_function == 'sigmoid':
            MIN = self.y.min()
            MAX = self.y.max()
            return y_pred * (MAX - MIN) + MIN
        else:
            # da implementare scaling per altre activations
            return NotImplemented

    def feedforward(self):
        """ """
        for i in range(self.n_layers):
            self.V[i] = np.dot(self.W[i], self.X_T if i == 0
                               else add_bias_mul(self.Y[i - 1]))
            self.Y[i] = activation_function(self.activation_function,
                                            self.V[i])

        total_instantaneous_error = list()
        instantaneous_error = (self.d - self.Y[-1])**2

        for column in range(instantaneous_error.shape[1]):
            tmp = 0
            for row in range(instantaneous_error.shape[0]):
                tmp += instantaneous_error[row][column]

            total_instantaneous_error.append(0.5 * tmp)

        self.empirical_risk.append(1/len(self.X) *
                                   sum(total_instantaneous_error))
        total_instantaneous_error = []

    def backpropagation(self, eta, epoch ):
        """

        Parameters
        ----------
        eta : the learning rate;

        epoch : the current epoch in which the backpropagation phase of the
                algorithm is execute, it is used during the momentum's
                application;

        alpha : the momentum constant;


        Returns
        -------

        """
        if epoch == 0:
            alpha = 0
        else:
            alpha = self.alpha
                
        for layer in reversed(range(self.n_layers)):
            # output layer
            if layer == self.n_layers - 1:
                self.delta[layer] = (self.d - self.Y[layer]) * \
                    activation_function(self.activation_function,
                                        self.V[layer],
                                        derivative=True)
                self.delta_W[layer] = (alpha * self.delta_W[layer])+\
                                      ((eta * self.delta[layer]).dot(self.Y[layer - 1].T))
            else:
                sum_tmp = self.delta[layer + 1].T.dot(self.W[layer + 1][:, 1:])
                self.delta[layer] = activation_function(
                    self.activation_function, self.V[layer],
                    derivative=True) * sum_tmp.T
                
                # input layer
                if layer == 0:
                    self.delta_W[layer] = (alpha * self.delta_W[layer])+\
                                          (eta * self.delta[layer].dot(self.X[:, 1:]))
                # hidden layers
                else:
                    self.delta_W[layer] = (alpha * self.delta_W[layer])+\
                                          (eta * self.delta[layer].dot(self.Y[layer - 1].T))
            # update weights
            self.W[layer][:, 1:] += self.delta_W[layer]

    def train(self, X, y, eta, alpha=0):
        """

        Parameters
        ----------
        X : the training set;

        y : the original target, also used in self.d as the internal (scaled)
            target;

        eta : the learning rate;

        alpha : the momentum constant, which default value represents the
                momentumless execution of the algorithm;


        Returns
        -------

        """
        self.X = X
        self.topology = compose_topology(self.X, self.hidden_sizes, y)

        self.X_T = add_bias_mul(X.T, axis=0)

        self.alpha = alpha
        self.eta = eta
       
        self.y = y
        self.d = self.target_scale(y)
        self.empirical_risk = list()

        print 'CREATED A ' + ' x '.join([str(i) for i in self.topology]) \
            + ' NEURAL NETWORK'

        self.init_weights()
        print 'STARTING WEIGHTS\n'
        for i in range(self.n_layers):
            print self.W[i]
            print '\n'

        for i in range(self.max_epochs):
            self.feedforward()
            self.backpropagation(eta, i)
            # TODO: stopping criteria

        print '\nFINAL WEIGHTS\n'
        for i in range(len(self.W)):
            print self.W[i]
            print '\n'

        print '\nSTARTING EMPIRICAL ERROR: {}\nCLOSING EMPIRICAL ERROR: {}'.\
            format(self.empirical_risk[0], self.empirical_risk[-1])

        # scaling back the output
        y_pred = self.Y[-1]
        # here rounding for classification
        self.y_pred = self.target_scale_back(y_pred)
