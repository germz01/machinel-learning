from __future__ import division

import numpy as np
from utils import activation_function

class NeuralNetwork(object):
    """Simple implementation of an Artificial Neural Network"""

    def __init__(self, hidden_sizes, activation='sigmoid',
                 max_epochs=1000, max_weight_init=0.7):

        #self.X = None
        #self.W = None
        #self.d = None

        self.hidden_sizes = hidden_sizes
        self.n_layers = len(hidden_sizes)+1 # considering the out layer
        #self.topology = None

        self.activation_function = activation
        self.V = [0 for i in range(self.n_layers)]
        self.Y = [0 for i in range(self.n_layers)]
        self.delta = [0 for i in range(self.n_layers)]
        self.delta_W = [0 for i in range(self.n_layers)]
        self.max_epochs = max_epochs
        self.max_weight_init = max_weight_init
        

    def init_weights(self):
        self.W = list()
        for i in range(1,len(self.topology)):
            self.W.append( np.random.uniform(
                -self.max_weight_init, self.max_weight_init,
                (self.topology[i], self.topology[i - 1] )))

    def init_weights_test(self):
        'weights init per testing' 
        self.W = list()
        for i in range(1,len(self.topology)):
            self.W.append( np.ones((self.topology[i], self.topology[i - 1])))


    def target_scale(self, y):
        if self.activation_function == 'sigmoid':
            MIN = y.min()
            MAX = y.max()
            return (y-MIN)/ (MAX-MIN)
        else:
            # da implementare scaling per altre activations
            return NotImplemented

    def target_scale_back(self, y_pred):
        if self.activation_function == 'sigmoid':
            MIN = self.y.min()
            MAX = self.y.max()
            return y_pred*(MAX-MIN)+MIN
        else:
            # da implementare scaling per altre activations
            return NotImplemented

    def feedforward(self):
        for i in range(self.n_layers):
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
        for layer in reversed(range(self.n_layers)):
            if layer == self.n_layers - 1:
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
                    self.delta_W[layer] = eta * self.delta[layer].dot(self.X)
                else:
                    self.delta_W[layer] = eta * self.delta[layer].dot(self.Y[layer-1].T)

        for i in range(self.n_layers):
            self.W[i] += self.delta_W[i]
    ## end backpropagation ########################################

    def train(self, X, y, eta):
        # inizialization
        self.X = X
        self.y = y # original target
        self.d = self.target_scale(y) # internal target
        self.empirical_risk = list()
        
        self.topology =  [X.shape[1]] + \
                         list(self.hidden_sizes) + \
                         [ 1 if len(y.shape)==1 else y.shape[1] ] #out size

        print 'CREATED A ' +  ' x '.join([str(i) for i in self.topology]) + ' NEURAL NETWORK'
        
        self.init_weights()

        print 'STARTING WEIGHTS\n'
        for i in range(self.n_layers):
            print self.W[i]
        ###########################################################

        for i in range(self.max_epochs):
            self.feedforward()
            self.backpropagation(eta)
            # TODO: stopping criteria

        print '\nFINAL WEIGHTS\n'
        for i in range(len(self.W)):
            print self.W[i]

        print '\nSTARTING EMPIRICAL ERROR: {}\nCLOSING EMPIRICAL ERROR: {}'.\
            format(self.empirical_risk[0], self.empirical_risk[-1])

        # scaling back the output
        y_pred = self.Y[-1]
        # here rounding for classification
        self.y_pred = self.target_scale_back(y_pred)

        
###########################################################
