from __future__ import division

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit


def add_bias_mul(X, axis=0):
    if axis == 0:
        # add ones on the first row
        return np.vstack((np.ones(X.shape[1]), X))
    else:
        # add ones on the first column
        tmp = np.ones(X.shape[0])
        tmp.shape = (X.shape[0], 1)

        return np.hstack((tmp, X))


def compose_topology(X, hidden_sizes, y):
    topology = [X.shape[1]] + list(hidden_sizes) + \
        [1 if len(y.shape) == 1 else y.shape[1]]

    return topology


def activation_function(func_name, x, derivative=False):
    if func_name == 'identity':
        return x
    elif func_name == 'sigmoid':
        if derivative:
            return expit(x) * (1. - expit(x))
        else:
            return expit(x)
    elif func_name == 'tanh':
        if derivative:
            return 1 - activation_function('tanh', x)**2
        else:
            return ((np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x)))

###########################################################
# plotting

def plot_learning_curve(empirical_risk, num_epochs,
                        fname = '../images/learning_curve.pdf'):
    plt.plot(range(num_epochs), empirical_risk)
    plt.title('LEARNING CURVE FOR A {} EPOCHS TRAINING PERIOD'.
              format(num_epochs))
    plt.xlabel('EPOCHS')
    plt.ylabel('EMPIRICAL RISK')
    initial_risk = mpatches.Patch(label='Initial E.R.: {}'.
                                  format(empirical_risk[0]))
    final_risk = mpatches.Patch(label='Final E.R.: {}'.
                                format(empirical_risk[-1]))
    plt.legend(handles=[initial_risk, final_risk])
    plt.grid()
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

###########################################################    
# error functions

def rmse(d, y):
        '''root mean square error'''
        p = d.shape[0]
        if len(d.shape) == 1:
                d = d.reshape((p,1))
        if len(y.shape) == 1:
               y  = y.reshape((p,1))

               return np.sqrt( np.einsum('pk->', (d-y)**2) /p)


def mee(d, y):
        ''' mean euclidean error'''
        p = d.shape[0]
        if len(d.shape) == 1:
                d = d.reshape((p,1))
        if len(y.shape) == 1:
               y  = y.reshape((p,1))

               return np.mean( np.sqrt(np.einsum('pk->p', (d-y)**2)) )

def mee_dev(d, y):
        ''' std. deviation for the euclidean error'''
        p = d.shape[0]
        if len(d.shape) == 1:
                d = d.reshape((p,1))
        if len(y.shape) == 1:
               y  = y.reshape((p,1))

               return np.std( np.sqrt(np.einsum('pk->p', (d-y)**2)) )
