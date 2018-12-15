from __future__ import division

import activations as act
import losses as lss
import numpy as np
import regularizers as reg
import utils as u

class NeuralNetwork(object):
    """ """
    def __init__(self, hidden_sizes, task='classifier'):

        self.hidden_sizes = hidden_sizes
        self.n_layers = len(hidden_sizes) + 1

        self.delta_W = [0 for i in range(self.n_layers)]
        self.delta_b = [0 for i in range(self.n_layers)]
        self.a = [0 for i in range(self.n_layers)]
        self.h = [0 for i in range(self.n_layers)]

    def set_weights(self):
        """
        This function initializes the network's weights matrices following
        the rule in Deep Learning, pag. 295

        Parameters
        ----------
        topology : a list of integer in which each number represents how many
                   neurons must to be added to the current layer

        Returns
        -------
        A list of matrices in which each matrix is a weights matrix
        """
        topology = self.topology
        W = []

        for i in range(1, len(topology)):
            low = - np.sqrt(6 / (topology[i - 1] + topology[i]))
            high = np.sqrt(6 / (topology[i - 1] + topology[i]))

            W.append(np.random.uniform(low, high, (topology[i],
                                                   topology[i - 1])))

        return W

    def get_weights(self):
        """
        This function returns the list containing the network's weights'
        matrices
        """
        for i in range(self.n_layers):
            print 'W{}: \n{}'.format(i, self.W[i])

    def set_bias(self):
        """
        This function initializes the bias for the neural network

        Parameters
        ----------
        topology : a list of integer in which each number represents how many
                   neurons must to be added to the current layer

        Returns
        -------
        A list of matrices in which each matrix is a bias matrix
        """
        b = []

        for i in range(1, len(self.topology)):
            b.append(np.random.uniform(-.7, .7, (self.topology[i], 1)))

        return b

    def get_bias(self):
        """
        This function returns the list containing the network's bias'
        matrices
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

        y : the target value, or array, for the record/batch given in input

        Returns
        -------
        The loss between the predicted output and the target output
        """
        for i in range(self.n_layers):
            self.a[i] = self.b[i] + (self.W[i].dot(x.reshape(-1, 1) if i == 0
                                                   else self.h[i - 1]))
            self.h[i] = act.A_F['sigmoid']['f'](self.a[i])

        return lss.mean_squared_error(self.h[-1], y)

    def back_propagation(self, x, y, eta):
        """
        This function implements the back propagation algorithm following
        Deep Learning, pag. 206

        Parameters
        ----------
        x : a record, or batch, from the dataset

        y : the target value, or target array, for the record/batch given in
            input

        eta : the learning rate

        Returns
        -------

        """
        g = lss.mean_squared_error(self.h[-1], y, gradient=True)

        for layer in reversed(range(self.n_layers)):
            g = np.multiply(g, act.A_F['sigmoid']['fdev'](self.a[layer]))

            self.delta_b[layer] = g

            # x.reshape(1, -1) ritorna x dentro un array in modo da farla
            # passare come una matrice
            self.delta_W[layer] = g.dot(self.h[layer - 1].T if layer != 0 else
                                        x.reshape(1, -1))

            g = self.W[layer].T.dot(g)

    def train(self, X, y, eta, alpha, regularizer, epochs):
        """
        This function traines the neural network whit the hyperparameters given
        in input

        Parameters
        ----------
        X : the dataset

        y : the target array

        eta : the learning rate

        alpha : the momentum constant

        regularizer : a list of two items, in which the first item represents
        the regularization constant and the second items represents the type
        of regularization, either L1 or L2, that has to be applied

        epochs : the (maximum) number of epochs for which the neural network
        has to be trained

        Returns
        -------

        """
        self.topology = u.compose_topology(X, self.hidden_sizes, y)
        self.epochs = epochs

        self.W = self.set_weights()
        self.b = self.set_bias()
        self.loss_online = []
        self.loss_epochs = []


        velocity_W = [0 for i in range(self.n_layers)]
        velocity_b = [0 for i in range(self.n_layers)]

        for e in range(epochs):
            for i in range(X.shape[0]):
                loss = self.forward_propagation(X[i], y[i])
                self.back_propagation(X[i], y[i], eta)

                for layer in range(self.n_layers):
                    weight_decay = reg.regularization(self.W[layer],
                                                      regularizer[0],
                                                      regularizer[1])

                    velocity_b[layer] = (alpha * velocity_b[layer]) - \
                        (eta * self.delta_b[layer])
                    self.b[layer] += velocity_b[layer]

                    velocity_W[layer] = (alpha * velocity_W[layer]) - \
                        (eta * (weight_decay + self.delta_W[layer]))
                    self.W[layer] += velocity_W[layer]

                self.loss_online.append(loss)
            self.loss_epochs.append(loss)

        print 'STARTED WITH LOSS {}, ENDED WITH {}'.format(self.loss_epochs[0], self.loss_epochs[-1])


if __name__ == '__main__':
    X = np.concatenate((np.random.normal(2., 1., (3, 2)),
                        np.random.normal(5., 1., (2, 2))),
                       axis=0)
    y = np.array([1, 1, 1, 0, 0]).reshape(5, 1)

    nn = NeuralNetwork([X.shape[1], 3, 1])
    nn.train(X, y, .1, .9, [0.01, 'l2'], 1000)
