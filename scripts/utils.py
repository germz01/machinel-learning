from __future__ import division

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# CONSTANTS

# This is the path for the directory in which the images are saved.
IMGS = '../images/'


def compose_topology(X, hidden_sizes, y):
    """
    This functions builds up the neural network's topology as a list.

    Parameters
    ----------
    X: numpy.ndarray
        the design matrix

    hidden_sizes: list
        a list of integers; every integer represents the number
        of neurons that will compose an hidden layer of the
        neural network

    y: numpy.ndarray
        the target column vector

    Returns
    -------
    A list of integers representing the neural network's topology.
    """
    topology = [X.shape[1]] + list(hidden_sizes) + \
        [1 if len(y.shape) == 1 else y.shape[1]]

    return topology

# PLOTTING RELATED FUNCTIONS


def plot_learning_curve(nn, fname='../images/learning_curve.pdf'):
    """
    This functions plots the learning curves for a model given in input.

    Parameters
    ----------
    nn: nn.NeuralNetwork
        the neural network

    fname: str
        where to save the learning curve's plot
        (Default value = '../images/learning_curve.pdf')

    Returns
    -------
    """

    par_str = r"""$\eta= {}, \alpha= {}, \lambda= {}$,'batch= {}, h-sizes={}""".format(
        np.round(nn.params['eta'], 2),
        np.round(nn.params['alpha'], 2),
        np.round(nn.params['reg_lambda'], 3),
        nn.params['batch_size'],
        nn.params['hidden_sizes']
    )

    plt.figure(figsize=(15, 5))
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
        accuracy=False,
        other_errors=None,
        accuracy_h_plot=None,
        accuracy_per_epochs=None,
        accuracy_per_epochs_va=None,
        figsize=None,
        fontsize_title=13,
        fontsize_labels=12,
        fontsize_info=12,
        fontsize_legend=12,
        MEE_VL=None,
        MEE_TR=None,
        stop_GL=None,
        stop_PQ=None
):
    """ Plots the learning curve with infos """

    # ###########################################################
    # legend info
    info = ''

    assert task in ('validation', 'testing')
    if task == 'validation':
        task_str = 'Validation'
        task_str_abbr = 'VL'
    elif task == 'testing':
        task_str = 'Test'
        task_str_abbr = 'TS'

    if accuracy:
        y_tr = np.array(error_per_epochs, dtype=np.float)*100
        y_va = np.array(error_per_epochs_va, dtype=np.float)*100

        # print 'Acc TR = {} %'.format(str(np.round(y_tr[-1], 1)))
        final_errors = [
            'Acc TR = {} %'.format(str(np.round(y_tr[-1], 1))),
            'Acc {} = {} %'.format(task_str_abbr,
                                   str(np.round(y_va[-1], 1)))
                                    ]
    else:
        y_tr = error_per_epochs
        y_va = error_per_epochs_va

        final_errors = [
            'MSE TR =' + str(np.round(y_tr[-1], 5)),
            'MSE {} ='.format(task_str_abbr) + str(np.round(y_va[-1], 5))
        ]
        if MEE_VL is not None:
            final_errors.extend([
                'MEE TR =' + str(np.round(MEE_TR, 5)),
                'MEE {} ='.format(task_str_abbr) + str(np.round(MEE_VL, 5))]
            )

    # appending other errors, ex: accuracy
    final_errors_str = '\n'.join(final_errors)
    if other_errors is not None:
        final_errors_str += other_errors
    if accuracy_h_plot:
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
    info += r'$\eta= {}$'.format(np.round(hyperparams['eta'], 6))+'\n'
    info += r'$\alpha= {}$'.format(np.round(hyperparams['alpha'], 2))+'\n'

    info += r'${}$ regularization'.format(
        'L_2' if hyperparams['reg_method'] == 'l2' else 'L_1') + '\n'
    info += r'$\lambda= {}$'.format(
        np.round(hyperparams['reg_lambda'], 3))

    info += '\n\nGD: {}'.format(hyperparams['batch_method'])+'\n'
    if hyperparams['batch_method'] != 'batch':
        info += 'mb={}\n'.format(hyperparams['batch_size'])

    info += '\nTopology:\n'
    info += '->'.join([str(el) for el in hyperparams['topology']])+'\n'
    info += '\nActivation: {}'.format(hyperparams['activation'][0])+'\n'

    ###########################################################
    plt.close()

    x_epochs = np.arange(len(error_per_epochs))

    if accuracy_h_plot is None:
        SMALL_SIZE = 11
        MEDIUM_SIZE = 16
        BIGGER_SIZE = 20

        plt.rc('font', size=MEDIUM_SIZE)
        plt.rc('axes', labelsize=BIGGER_SIZE)
        plt.rc('xtick', labelsize=MEDIUM_SIZE)
        plt.rc('ytick', labelsize=MEDIUM_SIZE)
        plt.rc('legend', fontsize=BIGGER_SIZE)
        # plt.rc('figure', titlesize=BIGGER_SIZE)
        plt.rc('axes', titlesize=BIGGER_SIZE)

        if figsize is None:
            figsize = (15, 7)
        plt.figure(figsize=figsize)
        grid = plt.GridSpec(1, 7, wspace=0.1, hspace=0.7, left=0.1,
                            top=0.9, bottom=0.15)

        plt.subplot(grid[0, :6])
        plt.plot(x_epochs, y_tr, label='Training', linestyle='-')
        plt.plot(x_epochs, y_va, label=task_str, linestyle='--')
        plt.xlabel('Epochs')
        # early stopping

        if stop_GL is not None:
            plt.axvline(stop_GL, linestyle=':', label='GL early stop')
            plt.axvline(np.argmin(error_per_epochs_va), linestyle='-', label='MSE VL min')
        if stop_PQ is not None:
            plt.axvline(stop_PQ, linestyle='-.', label='PQ early stop')
        if accuracy:
            plt.ylabel('Accuracy (%)')
        else:
            plt.ylabel('MSE')

        plt.title(title)
        plt.legend()
        plt.grid()

        plt.subplot(grid[0, 6:])

        plt.title('Info')
        plt.text(x=0.2, y=0.97, s=info,
                 ha='left', va='top', fontsize=MEDIUM_SIZE)

        plt.axis('off')

        plt.savefig(fname+'.png')
        plt.savefig(fname+'.pdf')

        plt.close()
    elif accuracy_h_plot:
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

        plt.savefig(fname, bbox_inches='tight')

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
