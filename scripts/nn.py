from __future__ import division

import activations as act
import losses as lss
import numpy as np
import regularizers as reg
import utils as u

from tqdm import tqdm


class NeuralNetwork(object):
    """ """
    def __init__(self, hidden_sizes, task='classifier'):

        self.hidden_sizes = hidden_sizes
        self.n_layers = len(hidden_sizes) + 1

        self.delta_W = [0 for i in range(self.n_layers)]
        self.delta_b = [0 for i in range(self.n_layers)]
        self.a = [0 for i in range(self.n_layers)]
        self.h = [0 for i in range(self.n_layers)]

    def set_weights(self, w_par=6):
        """
        This function initializes the network's weights matrices following
        the rule in Deep Learning, pag. 295

        Parameters
        ----------
        w_par : a parameter which is plugged into the formula for estimating
                the uniform interval for defining the network's weights
            (Default value = 6)

        Returns
        -------
        """
        W = []

        for i in range(1, len(self.topology)):
            low = - np.sqrt(w_par / (self.topology[i - 1] + self.topology[i]))
            high = np.sqrt(w_par / (self.topology[i - 1] + self.topology[i]))

            W.append(np.random.uniform(low, high, (self.topology[i],
                                                   self.topology[i - 1])))

        return W

    def get_weights(self):
        """
        This function returns the list containing the network's weights'
        matrices

        Parameters
        ----------

        Returns
        -------
        """
        for i in range(self.n_layers):
            print 'W{}: \n{}'.format(i, self.W[i])

    def set_bias(self):
        """
        This function initializes the bias for the neural network
        """
        b = []

        for i in range(1, len(self.topology)):
            b.append(np.random.uniform(-.7, .7, (self.topology[i], 1)))

        return b

    def get_bias(self):
        """
        This function returns the list containing the network's bias'
        matrices

        Parameters
        ----------

        Returns
        -------
        """
        for i in range(len(self.b)):
            print 'b{}: \n{}'.format(i, self.b[i])

    def forward_propagation(self, x, y):
        """
        This function implements the forward propagation algorithm following
        Deep Learning, pag. 205

        Parameters
        ----------
        x : a record, or batch, from the dataset

        y : the target array for the batch given in input


        Returns
        -------
        """
        for i in range(self.n_layers):
            self.a[i] = self.b[i] + (self.W[i].dot(x.T if i == 0
                                                   else self.h[i - 1]))
            self.h[i] = act.A_F['sigmoid']['f'](self.a[i])

        return lss.mean_squared_error(self.h[-1].T, y)

    def back_propagation(self, x, y):
        """
        This function implements the back propagation algorithm following
        Deep Learning, pag. 206

        Parameters
        ----------
        x : a record, or batch, from the dataset

        y : the target value, or target array, for the record/batch given in
            input

        Returns
        -------
        """
        g = lss.mean_squared_error(self.h[-1], y.T, gradient=True)

        for layer in reversed(range(self.n_layers)):
            g = np.multiply(g, act.A_F['sigmoid']['fdev'](self.a[layer]))
            # update bias, sum over patterns
            self.delta_b[layer] = g.sum(axis=1).reshape(-1, 1)

            # the dot product is summing over patterns
            self.delta_W[layer] = g.dot(self.h[layer - 1].T if layer != 0
                                        else x)
            # summing over previous layer units
            g = self.W[layer].T.dot(g)

    def train(self, X, y, eta, alpha=0, epochs=1000,
              batch_size=1, reg_lambda=0.0, reg_method='l2',
              regularizer=[0.0, 'l2'], w_par=6):
        """
        This function trains the neural network whit the hyperparameters given
        in input

        Parameters
        ----------
        X : the design matrix

        y : the target column vector

        eta : the learning rate

        regularizer : a list of two items, in which the first item represents
                      the regularization constant and the second items
                      represents the type of regularization, either L1 or L2,
                      that has to be applied

        alpha : the momentum constant
             (Default value = 0)

        epochs : the (maximum) number of epochs for which the neural network
                 has to be trained
             (Default value = 1000)

        batch_size : the batch size
             (Default value = 1)
        w_par : a parameter which is plugged into the formula for estimating
                the uniform interval for defining the network's weights
             (Default value = 6)

        Returns
        -------
        """
        self.topology = u.compose_topology(X, self.hidden_sizes, y)
        self.epochs = epochs
        self.X = X
        self.W = self.set_weights(w_par)
        self.b = self.set_bias()

        self.params = dict()
        self.params['eta'] = eta
        self.params['alpha'] = alpha
        self.params['batch_size'] = batch_size
        self.params['regularizer'] = regularizer
        self.params['hidden_sizes'] = self.hidden_sizes
        self.params['reg_method'] = reg_method
        self.params['reg_lambda'] = reg_lambda

        velocity_W = [0 for i in range(self.n_layers)]
        velocity_b = [0 for i in range(self.n_layers)]

        self.error_per_epochs = []
        self.error_per_batch = []

        for e in tqdm(range(epochs), desc='TRAINING'):
            error_per_batch = []

            dataset = np.hstack((X, y))
            np.random.shuffle(dataset)
            X, y = np.hsplit(dataset, [X.shape[1]])

            for b_start in np.arange(0, X.shape[0], batch_size):
                x_batch = X[b_start:b_start + batch_size, :]
                y_batch = y[b_start:b_start + batch_size, :]

                error = self.forward_propagation(x_batch, y_batch)
                self.error_per_batch.append(error)
                error_per_batch.append(error)

                self.back_propagation(x_batch, y_batch)

                for layer in range(self.n_layers):
                    weight_decay = reg.regularization(self.W[layer],
                                                      reg_lambda,
                                                      reg_method)

                    velocity_b[layer] = (alpha * velocity_b[layer]) \
                        - (eta / x_batch.shape[0]) * self.delta_b[layer]
                    self.b[layer] += velocity_b[layer]

                    velocity_W[layer] = (alpha * velocity_W[layer]) \
                        - ((eta / x_batch.shape[0])
                           * (weight_decay + self.delta_W[layer]))
                    self.W[layer] += velocity_W[layer]

            # summing up errors to compute overall MSE
            self.error_per_epochs.append(np.sum(error_per_batch)/X.shape[0])

        print 'STARTED WITH LOSS {}, ENDED WITH {}'.\
            format(self.error_per_epochs[0], self.error_per_epochs[-1])

    def predict(self, x, y):
        """

        Parameters
        ----------
        x :
        y :

        Returns
        -------

        """
        for layer in range(self.n_layers):
            self.a[layer] = self.W[layer].dot(x.T if layer == 0 else
                                              self.h[layer - 1])+self.b[layer]
            self.h[layer] = act.A_F['sigmoid']['f'](self.a[layer])

        return lss.mean_squared_error(self.h[-1].T, y)
        # return self.h[-1]

    def get_params(self):
        """
        Return the parameters of the nn instance

        Parameters
        ----------

        Returns
        -------
        params : dict
            parameters dictionary
        """
        return self.params
