from __future__ import division

import numpy as np
from utils import activation_function


class NeuralNetwork(object):
    """Simple implementation of an Artificial Neural Network"""

    def __init__(self, X, y, topology, function, max_epochs):
        print 'CREATING A {} x {} x {} NEURAL NETWORK'.format(topology[0],
                                                              topology[1],
                                                              topology[2])

        self.X = X
        self.W = self.generate_weights(topology)
        self.d = y
        self.activation_function = function
        self.V = [0 for i in range(len(self.W))]
        self.Y = [0 for i in range(len(self.W))]
        self.delta = [0 for i in range(len(self.W))]
        self.delta_W = [0 for i in range(len(self.W))]
        self.max_epochs = max_epochs
        self.empirical_risk = list()

    def generate_weights(self, topology):
        W = list()
        for i in range(1, len(topology)):
            # print 'FROM LAYER {} TO LAYER {}'.format(i - 1, i)
            w_m = np.random.uniform(-0.5, 0.5, (topology[i], topology[i - 1]))
            # print w_m
            W.append(w_m)

        return W

    def feedforward(self):
        for i in range(len(self.W)):
            self.V[i] = np.dot(self.W[i], self.X.T if i == 0 else
                               self.Y[i - 1])
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

    def backpropagation(self, eta):
        for layer in reversed(range(len(self.W))):
            if layer == len(self.W) - 1:
                self.delta[layer] = (self.d - self.Y[layer]) * \
                    activation_function(self.activation_function,
                                        self.V[layer],
                                        derivative=True)
                self.delta_W[layer] = ((eta * self.delta[layer]).dot(
                                   self.Y[layer - 1].T))
            else:
                sum_tmp = self.delta[layer + 1].T.dot(self.W[layer + 1])
                self.delta[layer] = activation_function(
                    self.activation_function, self.V[layer],
                    derivative=True) * sum_tmp.T

                if layer == 0:
                    self.delta_W[layer] = (eta * self.delta[layer]).dot(
                        self.X)
                else:
                    print 'DA IMPLEMENTARE'

        for i in range(len(self.W)):
            self.W[i] += self.delta_W[i]

    def train(self, eta):
        print 'STARTING WEIGHTS\n'
        for i in range(len(self.W)):
            print self.W[i]

        for i in range(self.max_epochs):
            self.feedforward()
            self.backpropagation(eta)

        print '\nFINAL WEIGHTS\n'
        for i in range(len(self.W)):
            print self.W[i]

        print '\nSTARTING EMPIRICAL ERROR: {}\nCLOSING EMPIRICAL ERROR: {}'.\
            format(self.empirical_risk[0], self.empirical_risk[-1])
