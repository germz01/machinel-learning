from __future__ import division

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit


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


def plot_learning_curve(empirical_risk, num_epochs, file_path = '../images/learning_curve.pdf'):
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
    plt.savefig( file_path, bbox_inches='tight')
    plt.close()
