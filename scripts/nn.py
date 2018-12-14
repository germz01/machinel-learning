from __future__ import division

import numpy as np
import regularizers as reg

from scipy.special import expit


def sigmoid(x, derivative=False):
    if derivative:
        return expit(x) * (1. - expit(x))
    else:
        return expit(x)


class NeuralNetwork(object):
    def __init__(self, topology):
        self.W = self.set_weights(topology)
        self.b = self.set_bias(topology)

        self.delta_W = [0 for i in range(len(self.W))]
        self.delta_b = [0 for i in range(len(self.W))]
        self.a = [0 for i in range(len(self.W))]
        self.h = [0 for i in range(len(self.W))]
        self.loss = []

    def set_weights(self, topology):
        W = []

        for i in range(1, len(topology)):
            low = - np.sqrt(6 / (topology[i - 1] + topology[i]))
            high = np.sqrt(6 / (topology[i - 1] + topology[i]))

            W.append(np.random.uniform(low, high, (topology[i],
                                                   topology[i - 1])))

        return W

    def get_weights(self):
        for i in range(len(self.W)):
            print 'W{}: \n{}'.format(i, self.W[i])

    def set_bias(self, topology):
        b = []

        for i in range(1, len(topology)):
            b.append(np.random.uniform(-.7, .7, (topology[i], 1)))

        return b

    def get_bias(self):
        for i in range(len(self.b)):
            print 'b{}: \n{}'.format(i, self.b[i])

    def forward_propagation(self, x, y):
        for i in range(len(self.W)):
            self.a[i] = self.b[i] + (self.W[i].dot(x.reshape(-1, 1) if i == 0
                                                   else self.h[i - 1]))
            self.h[i] = sigmoid(self.a[i])

        return 0.5 * np.sum(np.square(self.h[-1] - y))

    def back_propagation(self, x, y, eta):
        g = self.h[-1] - y

        for layer in reversed(range(len(self.W))):
            g = np.multiply(g, sigmoid(self.a[layer], derivative=True))

            self.delta_b[layer] = g

            # x.reshape(1, -1) ritorna x dentro un array in modo da farla
            # passare come una matrice
            self.delta_W[layer] = g.dot(self.h[layer - 1].T if layer != 0 else
                                        x.reshape(1, -1))

            g = self.W[layer].T.dot(g)

    def train(self, X, y, eta, alpha, regularizer, epochs):
        velocity_W = [0 for i in range(len(self.W))]
        velocity_b = [0 for i in range(len(self.W))]

        for e in range(epochs):
            for i in range(X.shape[0]):
                loss = self.forward_propagation(X[i], y[i])
                self.back_propagation(X[i], y[i], eta)

                for layer in range(len(self.W)):
                    weight_decay = reg.regularization(self.W[layer],
                                                      regularizer[0],
                                                      regularizer[1])

                    velocity_b[layer] = (alpha * velocity_b[layer]) - \
                        (eta * self.delta_b[layer])
                    self.b[layer] += velocity_b[layer]

                    velocity_W[layer] = (alpha * velocity_W[layer]) - \
                        (eta * (weight_decay + self.delta_W[layer]))
                    self.W[layer] += velocity_W[layer]

                self.loss.append(loss)

        print 'STARTED WITH LOSS {}, ENDED WITH {}'.format(self.loss[0],
                                                           self.loss[-1])


if __name__ == '__main__':
    X = np.concatenate((np.random.normal(2., 1., (3, 2)),
                        np.random.normal(5., 1., (2, 2))),
                       axis=0)
    y = np.array([1, 1, 1, 0, 0]).reshape(5, 1)

    nn = NeuralNetwork([X.shape[1], 3, 1])
    nn.train(X, y, .1, .9, [0.01, 'l2'], 1000)
