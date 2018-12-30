from __future__ import division

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict

# CONSTANTS

# This is the path for the directory in which the images are saved.
IMGS = '../images/'


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


def from_dict_to_list(grid):
    to_ret = defaultdict(list)

    for record in grid:
        for parameter, value in record.items():
            to_ret[parameter].append(value)

    return to_ret

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


def plot_error(nn, fname='../images/learning_curve.pdf'):
    """ plotting learning curve """

    par_str = r"""$\eta= {}, \alpha= {}, \lambda= {}$,'batch= {}, h-sizes={}""".format(
        np.round(nn.params['eta'], 2),
        np.round(nn.params['alpha'], 2),
        np.round(nn.params['reg_lambda'], 3),
        nn.params['batch_size'],
        nn.params['hidden_sizes']
    )

    plt.plot(range(len(nn.error_per_epochs)), nn.error_per_epochs)
    plt.ylabel('MSE error by epoch')
    plt.xlabel('Epochs')
    plt.grid()
    plt.suptitle('Learning curve')
    plt.title(par_str, fontsize=10)
    plt.savefig(fname)
    plt.tight_layout()
    plt.close()


def binarize_attribute(attribute, n_categories):
    """
    Binarize a vector of categorical values

    Parameters
    ----------
    attribute : numpy.ndarray or list
         numpy array with shape (p,1) or (p,) or list, containing
         categorical values.

    n_categories : int
        number of categories.
    Returns
    -------
    bin_att : numpy.ndarray
        binarized numpy array with shape (p, n_categories)
    """
    n_patterns = len(attribute)
    bin_att = np.zeros((n_patterns, n_categories), dtype=int)
    for p in range(n_patterns):
        bin_att[p, attribute[p]-1] = 1

    return bin_att


def binarize(X, categories_sizes):
    """
    Binarization of the dataset XWhat it does?

    Parameters
    ----------
    X : numpy.darray
        dataset of categorical values to be binarized.

    categories_sizes : list
        number of categories of each X column

    Returns
    -------
    out : numpy.darray
        Binarized dataset
    """

    atts = list()
    for col in range(X.shape[1]):
        atts.append(binarize_attribute(X[:, col], categories_sizes[col]))

    # h stack of the binarized attributes
    out = atts[0]
    for att in atts[1:]:
        out = np.hstack([out, att])

    return out
