from __future__ import division

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from scipy.special import expit

# CONSTANTS

# This is the path for the directory in which the images are saved.
IMGS = '../images/'

# UTILITIES RELATED FUNCTIONS

# ACTS = {'activation_name': {'f':f(), 'fdev':fdev(), 'range': (a,b) }}

ACTS = {
    'identity':
    {
        'f': lambda x: x,
        'fdev': lambda x: 1,
        'range': (np.NINF, np.Inf)
    },
    'sigmoid':
    {
        'f': lambda x: expit(x),
        'fdev': lambda x: expit(x) * (1. - expit(x)),
        'range' : (0,1)
    },
    'tanh':
    {
        'f': lambda x: np.tanh(x),
        'fdev': lambda x: 1 - np.tanh(x)**2,
        'range': (-1,-1)
    },
    'relu':
    {
        'f': lambda x: 0 if x < 0 else x,
        'fdev': lambda x: 0 if x < 0 else 1,
        'range': (0, np.Inf)
    }
}


# This dictionary contains the intervals in which the activation functions
# defined in the activation_function function are defined.
ACT_FUNC_INT = {
    'identity': [np.NINF, np.Inf],
    'sigmoid': [0, 1],
    'tanh': [-1, 1],
    'relu': [0, np.Inf]
}

def activation_function(func_name, x, derivative=False):
    """
    Apply an activation function, or its derivative, to a matrix/array given
    in input.

    Parameters
    ----------
    func_name : the name of the activation function

    x : the input to be submitted to the activation function

    derivative :
         (Default value = False)
         wheter or not to use the derivative of the activation function

    Returns
    -------
    The result of applying the activation function to the input x.
    """
    if func_name == 'identity':
        return 1 if derivative else x
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
    elif func_name == 'relu':
        if derivative:
            return 0 if x < 0 else 1
        else:
            return 0 if x < 0 else x


def add_bias_mul(X, axis=0):
    """
    Adds a row, or column, of 1s to a matrix which is given in input.

    Parameters
    ----------
    X : the matrix in which the row (or column) of 1s has to be added;

    axis :
         (Default value = 0)
         0 means 'row' and 1 means 'column';

    Returns
    -------
    A new matrix.
    """
    if axis == 0:
        return np.vstack((np.ones(X.shape[1]), X))
    else:
        tmp = np.ones(X.shape[0])
        tmp.shape = (X.shape[0], 1)

        return np.hstack((tmp, X))


def compose_topology(X, hidden_sizes, y):
    """

    Parameters
    ----------
    X : the matrix representing the dataset

    hidden_sizes : a list of integers. Every integer represents the number
                   of neurons that will compose an hidden layer of the
                   neural network;

    y : a list containing the targets for the dataset X;


    Returns
    -------
    A list of integers representing the neural network's topology.
    """
    topology = [X.shape[1]] + list(hidden_sizes) + \
        [1 if len(y.shape) == 1 else y.shape[1]]

    return topology

# PLOTTING RELATED FUNCTIONS


def plot_learning_curve(stats, num_epochs, momentum,
                        fname='../images/learning_curve.pdf'):
    """
    This function is used to plot the learning curve for all the errors
    collected during the network's training phase.

    Parameters
    ----------
    stats : a list which contains the error collected during the network's
            training phase;

    num_epochs : the epochs of training;

    momentum : the type of momentum selected for the training, either classic
               or nesterov;

    fname :
         (Default value = '../images/learning_curve.pdf')

    Returns
    -------
    The errors' plots.
    """
    for i in range(len(stats)):
        plt.plot(range(num_epochs), stats[i])
        plt.title('LEARNING CURVE FOR A {} EPOCHS TRAINING PERIOD'.
                  format(num_epochs))
        plt.xlabel('EPOCHS')
        plt.ylabel('EMPIRICAL RISK')
        initial_risk = mpatches.Patch(label='Initial E.R.: {}'.
                                      format(stats[i][0]))
        final_risk = mpatches.Patch(label='Final E.R.: {}'.
                                    format(stats[i][-1]))
        plt.legend(handles=[initial_risk, final_risk])
        plt.grid()
        plt.savefig(IMGS + fname[i] + momentum + '.png' if len(fname) != 1
                    else fname, bbox_inches='tight')
        plt.close()


# ERROR RELATED FUNCTIONS


def mee(d, y):
    """mean euclidean error

    Parameters
    ----------
    d :

    y :


    Returns
    -------

    """
    p = d.shape[0]
    if len(d.shape) == 1:
        d = d.reshape((p, 1))
    if len(y.shape) == 1:
        y = y.reshape((p, 1))

        return np.mean(np.sqrt(np.einsum('pk->p', (d-y)**2)))


def mee_dev(d, y):
    """std. deviation for the euclidean error

    Parameters
    ----------
    d :

    y :


    Returns
    -------

    """
    p = d.shape[0]
    if len(d.shape) == 1:
        d = d.reshape((p, 1))
    if len(y.shape) == 1:
        y = y.reshape((p, 1))

        return np.std(np.sqrt(np.einsum('pk->p', (d-y)**2)))


def rmse(d, y):
    """root mean square error

    Parameters
    ----------
    d :

    y :


    Returns
    -------

    """
    p = d.shape[0]
    if len(d.shape) == 1:
        d = d.reshape((p, 1))
    if len(y.shape) == 1:
        y = y.reshape((p, 1))

        return np.sqrt(np.einsum('pk->', (d-y)**2) / p)
