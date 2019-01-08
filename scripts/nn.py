from __future__ import division

import activations as act
import losses as lss
import numpy as np
import regularizers as reg
import utils as u

from tqdm import tqdm


class NeuralNetwork(object):
    """
    This class represents an implementation for a simple neural network.

    Attributes
    ----------
    TODO
    """
    def __init__(self, X, y, hidden_sizes=[10],
                 eta=0.5, alpha=0, epochs=1000,
                 batch_size=1, reg_lambda=0.0, reg_method='l2',
                 w_par=6, activation='sigmoid', task='classifier'):
        """
        The class' constructor.

        Parameters
        ----------
        X: numpy.ndarray
            the design matrix

        y: numpy.ndarray
            the target column vector

        hidden_sizes: list
            a list of integers. The list's length represents the number of
            neural network's hidden layers and each integer represents the
            number of neurons in a hidden layer

        eta : float
            the learning rate
            (Default value = 0.5)

        alpha : float
            the momentum constant
            (Default value = 0)

        epochs : int
            the (maximum) number of epochs for which the neural network
            has to be trained
            (Default value = 1000)

        batch_size : int
            the batch size
            (Default value = 1)

        reg_lambda: float
            the regularization factor
            (Default value = 0.0)

        reg_method: str
            the regularization method, either l1 or l2 regularization are
            availables
            (Default value = 'l2')

        w_par: int
            the parameter for initializing the network's weights matrices
            following the rule in Deep Learning, pag. 295
            (Default value = 6)

        activation: str
            the activation function to use for each layer, either
            'sigmoid', 'relu', 'tanh', 'identity'
            (Default value = 'sigmoid')

        task: str
            the task that the neural network has to perform, either
            'classifier' or 'regression'
            (Default value = 'classifier')

        Returns
        -------
        """

        self.hidden_sizes = hidden_sizes
        self.n_layers = len(hidden_sizes) + 1
        self.topology = u.compose_topology(X, self.hidden_sizes, y)

        self.X = X

        self.eta = eta
        self.alpha = alpha
        self.batch_size = batch_size
        self.reg_method = reg_method
        self.reg_lambda = reg_lambda
        self.epochs = epochs
        self.activation = activation

        self.W = self.set_weights(w_par)
        self.W_copy = [w.copy() for w in self.W]
        self.b = self.set_bias()
        self.b_copy = [b.copy() for b in self.b]

        self.params = self.get_params()

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
        self.params = dict()
        self.params['eta'] = self.eta
        self.params['alpha'] = self.alpha
        self.params['batch_size'] = self.batch_size
        self.params['hidden_sizes'] = self.hidden_sizes
        self.params['reg_method'] = self.reg_method
        self.params['reg_lambda'] = self.reg_lambda
        self.params['epochs'] = self.epochs
        self.params['activation'] = self.activation

        return self.params

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
            self.h[i] = act.A_F[self.activation]['f'](self.a[i])

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
            g = np.multiply(g, act.A_F[self.activation]['fdev'](self.a[layer]))
            # update bias, sum over patterns
            self.delta_b[layer] = g.sum(axis=1).reshape(-1, 1)

            # the dot product is summing over patterns
            self.delta_W[layer] = g.dot(self.h[layer - 1].T if layer != 0
                                        else x)
            # summing over previous layer units
            g = self.W[layer].T.dot(g)

    def train(self, X, y):
        """
        This function trains the neural network whit the hyperparameters given
        in input

        Parameters
        ----------
        X : numpy.ndarray
            the design matrix

        y : numpy.ndarray
            the target column vector


        Returns
        -------
        """
        velocity_W = [0 for i in range(self.n_layers)]
        velocity_b = [0 for i in range(self.n_layers)]

        self.error_per_epochs = []
        self.error_per_batch = []

        for e in tqdm(range(self.epochs), desc='TRAINING'):
            error_per_batch = []

            dataset = np.hstack((X, y))
            np.random.shuffle(dataset)
            X, y = np.hsplit(dataset, [X.shape[1]])

            for b_start in np.arange(0, X.shape[0], self.batch_size):
                x_batch = X[b_start:b_start + self.batch_size, :]
                y_batch = y[b_start:b_start + self.batch_size, :]

                error = self.forward_propagation(x_batch, y_batch)
                self.error_per_batch.append(error)
                error_per_batch.append(error)

                self.back_propagation(x_batch, y_batch)

                for layer in range(self.n_layers):
                    weight_decay = reg.regularization(self.W[layer],
                                                      self.reg_lambda,
                                                      self.reg_method)

                    velocity_b[layer] = (self.alpha * velocity_b[layer]) \
                        - (self.eta / x_batch.shape[0]) * self.delta_b[layer]
                    self.b[layer] += velocity_b[layer]

                    velocity_W[layer] = (self.alpha * velocity_W[layer]) \
                        - ((self.eta / x_batch.shape[0])
                           * (weight_decay + self.delta_W[layer]))
                    self.W[layer] += velocity_W[layer]

            # summing up errors to compute overall MSE
            self.error_per_epochs.append(np.sum(error_per_batch)/X.shape[0])

        # print 'STARTED WITH LOSS {}, ENDED WITH {}'.\
        #     format(self.error_per_epochs[0], self.error_per_epochs[-1])

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
            self.h[layer] = act.A_F[self.activation]['f'](self.a[layer])

        return lss.mean_squared_error(self.h[-1].T, y)
        # return self.h[-1]

    def reset(self):
        """
        This function is used in order to reset the neural network inner
        variables. It is mainly used during the validation process.

        Parameters
        ----------

        Returns
        -------
        """
        self.W = [w.copy() for w in self.W_copy]
        self.b = [b.copy() for b in self.b_copy]
        self.delta_W = [0 for i in range(self.n_layers)]
        self.delta_b = [0 for i in range(self.n_layers)]
        self.a = [0 for i in range(self.n_layers)]
        self.h = [0 for i in range(self.n_layers)]
