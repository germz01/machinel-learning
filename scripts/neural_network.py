from __future__ import division

import numpy as np
from utils import add_bias_mul
from utils import compose_topology
import utils as u


class NeuralNetwork(object):
    """Implementation of an Artificial Neural Network"""

    def __init__(self, hidden_sizes, task='classifier', activation=['sigmoid'],
                 max_epochs=1000, max_weight_init=0):
        self.hidden_sizes = hidden_sizes
        self.n_layers = len(hidden_sizes) + 1

        self.V = [0 for i in range(self.n_layers)]
        self.Y = [0 for i in range(self.n_layers)]
        self.delta = [0 for i in range(self.n_layers)]
        self.delta_W = [0 for i in range(self.n_layers)]
        self.max_epochs = max_epochs
        self.max_weight_init = max_weight_init

        self.activation = activation
        if len(activation) == 1:
            self.acts = [u.ACTS[activation[0]] for l in range(self.n_layers)]
        else:
            self.acts = [u.ACTS[activation[i]] for i in activation]

        if task == 'regression':
            self.acts[-1] = u.ACTS['identity']['f']
            self.acts_dev[-1] = u.ACTS['identity']['fdev']

    def init_weights(self):
        """
        Initilize the network's weights using either the interval given
        during the creation of the network or activation function's one.
        """
        self.W = list()
        interval = self.acts[0]['range'] if \
            self.max_weight_init == 0 else \
            [- self.max_weight_init, self.max_weight_init]

        for i in range(1, len(self.topology)):
            self.W.append(np.random.uniform(interval[0], interval[1],
                          (self.topology[i], self.topology[i - 1] + 1)))

    def target_scale(self, y):
        """

        Parameters
        ----------
        y :


        Returns
        -------

        """
        if self.activation[0] == 'sigmoid':
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
        if self.activation[0] == 'sigmoid':
            MIN = self.y.min()
            MAX = self.y.max()
            return y_pred * (MAX - MIN) + MIN
        else:
            # da implementare scaling per altre activations
            return NotImplemented

    def feedforward(self):
        """
        The feedforward phase of the backpropagation algorithm.
        """
        for l in range(self.n_layers):
            self.V[l] = np.dot(self.W[l], self.X_T if l == 0
                               else add_bias_mul(self.Y[l - 1]))
            self.Y[l] = self.acts[l]['f'](self.V[l])

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

        # empirical_risk = rmse**2 /2
        self.error_rmse.append(u.rmse(self.Y[-1].T, self.d))
        self.error_mee.append(u.mee(self.Y[-1].T, self.d))
        self.error_mee_dev.append(u.mee_dev(self.Y[-1].T, self.d))

    def backpropagation(self, eta, epoch):
        """
        The backpropagation phase of the backpropagation algorithm.

        Parameters
        ----------
        eta : the learning rate;

        epoch : the current epoch in which the backpropagation phase of the
                algorithm is execute, it is used during the momentum's
                application;

        Returns
        -------

        """
        if epoch == 0:
            alpha = 0
        else:
            alpha = self.alpha

        for layer in reversed(range(self.n_layers)):
            # APPLICATION OF THE NESTEROV'S MOMENTUM IF REQUIRED
            if self.momentum == 'nesterov':
                self.W[layer] += alpha * self.delta_W[layer]

            # DELTA COMPUTATION
            if layer == self.n_layers - 1:
                # OUTPUT LAYER
                self.delta[layer] = (self.d - self.Y[layer]) * \
                    self.acts[layer]['fdev'](self.V[layer])

            else:
                # HIDDEN LAYERS
                sum_tmp = self.delta[layer + 1].T.dot(self.W[layer + 1][:, 1:])
                self.delta[layer] = self.acts[layer]['fdev'](self.V[layer]) * \
                    sum_tmp.T

            # GENERALIZED DELTA RULE
            self.delta_W[layer] = (alpha * self.delta_W[layer]) + \
                                  (eta * self.delta[layer].
                                      dot(self.X_T.T if layer == 0 else
                                          add_bias_mul(self.Y[layer - 1].T,
                                                       axis=1)))

        # weights update
        for layer in range(self.n_layers):
            self.W[layer] += self.delta_W[layer]

    def train(self, X, y, eta, momentum='classic', alpha=0,
              epsilon=0):
        """
        A wrapper function that is used in order to train the network.

        Parameters
        ----------
        X : the training set;

        y : the original target, also used in self.d as the internal (scaled)
            target;

        eta : the learning rate;

        momentum : the momentum's type that will be applied during the
                   backpropagation phase. Is either 'classic' or 'nesterov';

        alpha : the momentum constant, which default value represents the
                momentumless execution of the algorithm;

        epsilon: the threshold that enables the early stopping of the
                 neural network's training;


        Returns
        -------

        """
        self.X = X
        self.X_T = add_bias_mul(X.T, axis=0)
        self.topology = compose_topology(self.X, self.hidden_sizes, y)
        self.momentum = momentum
        self.alpha = alpha
        self.eta = eta
        self.epsilon = epsilon
        self.y = y
        self.d = self.target_scale(y)

        self.empirical_risk = list()
        self.error_rmse = list()
        self.error_mee = list()
        self.error_mee_dev = list()

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

            diff = 0 if i == 0 else self.empirical_risk[i - 1] - \
                self.empirical_risk[i]

            # EARLY STOPPING ENABLED ONLY IF EPSILON != 0
            if epsilon != 0 and i != 0 and diff < self.epsilon:
                print 'EARLY STOPPING AT EPOCH {}'.format(i)
                break

        print '\nFINAL WEIGHTS\n'
        for i in range(len(self.W)):
            print self.W[i]
            print '\n'

        print '\nSTARTING EMPIRICAL ERROR: {}\nCLOSING EMPIRICAL ERROR: {}'.\
            format(self.empirical_risk[0], self.empirical_risk[-1])

        # scaling back the output
        y_pred = self.Y[-1]
        # here rounding for classification
        if self.activation[0] == 'sigmoid':
            # da sistemare target_scale
            self.y_pred = self.target_scale_back(y_pred)

    def predict(self, X_test):
        """
        This function is used in order to give a prediction on a test set given
        in input.

        Parameters
        ----------
        X_test : the test set;

        Returns
        -------
        The prediction made by the network's output layer.
        """
        self.V_pred = [0 for i in range(self.n_layers)]
        self.Y_pred = [0 for i in range(self.n_layers)]

        for l in range(self.n_layers):
            self.V_pred[l] = np.dot(self.W[l],
                                    add_bias_mul(X_test.T) if l == 0
                                    else add_bias_mul(self.Y_pred[l - 1]))
            self.Y_pred[l] = self.acts[l](self.V_pred[l])

        return self.target_scale_back(self.Y_pred[-1])
