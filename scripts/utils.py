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
    """
    This function is used in order to convert an HyperGrid or HyperRandomGrid
    object in a dictionary in which each key is an hyperparameter's name and
    each value is an array of possible values for that hyperparameter.

    Parameters
    ----------
    grid: HyperGrid or HyperRandomGrid
        the grid object to convert

    Returns
    -------
    A dictionary. Each one of the dictionary's keys is a hyperparameter name,
    i.e 'eta', alpha,..., and the corresponding value is an array of possible
    value for that hyperparameter.
    """
    to_ret = defaultdict(list)

    for record in grid:
        for parameter, value in record.items():
            to_ret[parameter].append(value)

    return to_ret

# PLOTTING RELATED FUNCTIONS


def plot_learning_curve_old(stats, num_epochs, momentum,
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


def plot_learning_curve(nn, fname='../images/learning_curve.pdf'):
    """ plotting learning curve """

    par_str = r"""$\eta= {}, \alpha= {}, \lambda= {}$,'batch= {}, h-sizes={}""".format(
        np.round(nn.params['eta'], 2),
        np.round(nn.params['alpha'], 2),
        np.round(nn.params['reg_lambda'], 3),
        nn.params['batch_size'],
        nn.params['hidden_sizes']
    )

    plt.plot(range(len(nn.error_per_epochs)),
             nn.error_per_epochs, linestyle='--',
             label='training')
    if nn.error_per_epochs_va is not None:
        plt.plot(range(len(nn.error_per_epochs_va)),
                 nn.error_per_epochs_va, linestyle='-',
                 label='validation')

    if nn.stop_GL is not None:
        plt.axvline(nn.stop_GL, linestyle=':', label='GL early stop')

    if nn.stop_PQ is not None:
        plt.axvline(nn.stop_PQ, linestyle='-.', label='PQ early stop')

    plt.ylabel('MSE error by epoch')
    plt.xlabel('Epochs')
    plt.grid()
    plt.suptitle('Learning curve')
    plt.legend()
    plt.title(par_str, fontsize=10)
    plt.savefig(fname)
    plt.tight_layout()
    plt.close()


def plot_learning_curve_info(
        error_per_epochs, error_per_epochs_va,
        hyperparams,
        fname,
        task,
        title='Learning Curve',
        labels=None,
        other_errors=None,
        accuracy_plot=None,
        accuracy_per_epochs=None,
        accuracy_per_epochs_va=None,
        figsize=(10, 6),
        fontsize_title=13,
        fontsize_labels=12,
        fontsize_info=12,
        fontsize_legend=12):
    """ Plots the learning curve with infos """

    x_epochs = np.arange(len(error_per_epochs))
    y_tr = error_per_epochs
    y_va = error_per_epochs_va

    # ###########################################################
    # legend info
    info = ''

    assert task in ('validation', 'testing')
    if task == 'validation':
        task_str = 'Validation'
        task_str_abbr = 'VA'
    elif task == 'testing':
        task_str = 'Testing'
        task_str_abbr = 'TS'

    final_errors = [
        'MSE TR =' + str(np.round(y_tr[-1], 5)),
        'MSE {} ='.format(task_str_abbr) + str(np.round(y_va[-1], 5))
    ]
    # appending other errors, ex: accuracy
    final_errors_str = '\n'.join(final_errors)
    if other_errors is not None:
        final_errors_str += other_errors
    if accuracy_plot:
        acc_errors = [
            'Acc TR = {} %'.format(np.round(accuracy_per_epochs[-1]*100),1),
            'Acc {} = {} %'.format(task_str_abbr,
                                   np.round(accuracy_per_epochs_va[-1]*100,1))
        ]
        acc_errors_str = '\n'.join(acc_errors)+'\n'
        final_errors_str += '\n'+acc_errors_str

    info += '\nFinal Errors:' + '\n'
    info += final_errors_str + '\n'

    # hyperparameters string
    info += '\nHyperparameters:' + '\n'
    info += r'$\eta= {}$'.format(np.round(hyperparams['eta'], 2))+'\n'
    info += r'$\alpha= {}$'.format(np.round(hyperparams['alpha'], 2))+'\n'

    info += r'${}$ regularization'.format(
        'L_2' if hyperparams['reg_method'] == 'l2' else 'L_1') + '\n'
    info += r'$\lambda= {}$'.format(
        np.round(hyperparams['reg_lambda'], 3))

    info += '\nGD: {}'.format(hyperparams['batch_method'])+'\n'
    if hyperparams['batch_method'] != 'batch':
        info += 'mb={}\n'.format(hyperparams['batch_size'])

    info += '\nTopology:\n'
    info += '->'.join([str(el) for el in hyperparams['topology']])+'\n'
    info += '\nActivation: {}'.format(hyperparams['activation'][0])+'\n'

    ###########################################################
    plt.close()

    if accuracy_plot is None:
        plt.figure(figsize=figsize)
        grid = plt.GridSpec(1, 4, wspace=0.1, hspace=0.3, left=0.1)

        plt.subplot(grid[0, :3])
        plt.plot(x_epochs, y_tr, label='Training', linestyle='-')
        plt.plot(x_epochs, y_va, label=task_str, linestyle='--')

        plt.xlabel('Epochs', fontsize=fontsize_labels)
        plt.ylabel('MSE', fontsize=fontsize_labels)
        plt.title(title, fontsize=fontsize_title)
        plt.legend(fontsize=fontsize_legend)
        plt.grid()

        plt.subplot(grid[0, 3:])

        plt.title('Info')
        plt.text(x=0., y=0.97, s=info,
                 ha='left', va='top', fontsize=11)

        plt.axis('off')

        plt.savefig(fname)

        plt.close()
    elif accuracy_plot:
        SMALL_SIZE = 11
        MEDIUM_SIZE = 12
        BIGGER_SIZE = 14

        plt.rc('font', size=SMALL_SIZE)
        plt.rc('axes', titlesize=SMALL_SIZE)
        plt.rc('axes', labelsize=MEDIUM_SIZE)
        plt.rc('xtick', labelsize=SMALL_SIZE)
        plt.rc('ytick', labelsize=SMALL_SIZE)
        plt.rc('legend', fontsize=SMALL_SIZE)
        plt.rc('figure', titlesize=BIGGER_SIZE)
        plt.rc('axes', titlesize=BIGGER_SIZE)
        # plt.rc('title', titlesize=BIGGER_SIZE)

        figsize = (10, 4)
        plt.figure(figsize=figsize)
        grid = plt.GridSpec(1, 5, wspace=0.7, left=0.1, bottom=0.2)

        # MSE plot
        plt.subplot(grid[0, :2])
        plt.plot(x_epochs, y_tr, label='Training', linestyle='-')
        plt.plot(x_epochs, y_va, label='Validation', linestyle='--')

        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.title(title)
        plt.legend()
        plt.grid()
        # Accuracy plot
        plt.subplot(grid[0, 2:4])
        plt.plot(x_epochs,
                 np.array(accuracy_per_epochs, dtype=np.float)*100,
                 label='Training', linestyle='-')
        plt.plot(x_epochs,
                 np.array(accuracy_per_epochs_va, dtype=np.float)*100,
                 label=task_str, linestyle='--')

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title(title)
        plt.legend()
        plt.grid()

        plt.subplot(grid[0, 4:])

        plt.title('Info')
        plt.text(x=0., y=1, s=info,
                 ha='left', va='top')
                 # bbox={'capstyle': 'round', 'fill': False})

        plt.axis('off')

        plt.savefig(fname)

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
