import numpy as np
import pandas as pd

import nn as NN
import utils as u
import validation as valid
import metrics

import time
import json
import os.path
import imp

from pprint import pprint


imp.reload(u)
imp.reload(NN)
###########################################################
# LOADING DATASET

fpath = '../data/monks/'

names = ['monks-1_train',
         'monks-1_test',
         'monks-2_train',
         'monks-2_test',
         'monks-3_train',
         'monks-3_test']

datasets = {name: pd.read_csv(fpath+name+'_bin.csv').values
            for name in names}

###########################################################


def final_training(dataset, trials, hyperparams, info='', epochs_plot=None):
    ''' Final training for Monks'''

    design_set = datasets['monks-{}_train'.format(dataset)]
    test_set = datasets['monks-{}_test'.format(dataset)]

    y_design, X_design = np.hsplit(design_set, [1])
    y_test, X_test = np.hsplit(test_set, [1])

    # simmetrized X_design:
    X_design = (X_design*2-1)
    X_test = (X_test*2-1)
    design_set = np.hstack((y_design, X_design))
    test_set = np.hstack((y_test, X_test))

    fpath = '../data/monks/results/monks_{}/img_final/'.format(dataset)

    for trial in range(trials):
        nn = NN.NeuralNetwork(
            X_design, y_design, **hyperparams
        )
        nn.train(X_design, y_design, X_test, y_test)

        if epochs_plot is None:
            epochs_plot = len(nn.error_per_epochs)

        u.plot_learning_curve_info(
            error_per_epochs=nn.error_per_epochs[:epochs_plot],
            error_per_epochs_va=nn.error_per_epochs_va[:epochs_plot],
            hyperparams=nn.get_params(),
            task='testing',
            title='Monks-{}, MSE Learning Curve'.format(dataset),
            fname=fpath+'monks_{}_{}MSE_final_{:02d}'.format(
                dataset, info, trial))

        u.plot_learning_curve_info(
            error_per_epochs=nn.accuracy_per_epochs[:epochs_plot],
            error_per_epochs_va=nn.accuracy_per_epochs_va[:epochs_plot],
            hyperparams=nn.get_params(),
            task='testing',
            accuracy=True,
            title='Monks-{}, Accuracy Learning Curve'.format(dataset),
            fname=fpath+'monks_{}_{}ACC_final_{:02d}'.format(
                dataset, info, trial))

    # return X_design, X_test


###########################################################

# MONK-1

hyperparams_1 = dict(
        eta=0.8,
        hidden_sizes=[3],
        alpha=0.9,
        reg_method='l2', reg_lambda=0.0,
        epochs=1000,
        batch_size='batch',  # 'batch',
        activation='sigmoid',
        task='classifier',
        # early_stop='testing',  # 'testing',
        epsilon=5,
        # w_method='DL',
        # w_par=6,
        w_method='uniform',
        w_par=1./17
)

final_training(dataset=1, trials=20, hyperparams=hyperparams_1)

###########################################################

# MONK-2

hyperparams_2 = dict(
        eta=0.5,
        hidden_sizes=[3],
        alpha=0.9,
        reg_method='l2', reg_lambda=0.0,
        epochs=1000,
        batch_size='batch',  # 'batch',
        activation='sigmoid',
        task='classifier',
        # early_stop='testing',  # 'testing',
        epsilon=5,
        # w_method='DL',
        # w_par=6,
        w_method='uniform',
        w_par=1./17
)

# final_training(dataset=2, trials=20, hyperparams=hyperparams_2)

###########################################################

# MONK-3

# without regularization
hyperparams_3 = dict(
        eta=0.6,
        hidden_sizes=[3],
        alpha=0.9,
        reg_method='l2', reg_lambda=0.0,
        epochs=1000,
        batch_size='batch',  # 'batch',
        activation='sigmoid',
        task='classifier',
        # early_stop='testing',  # 'testing',
        epsilon=5,
        # w_method='DL',
        # w_par=6,
        w_method='uniform',
        w_par=1./17
)

# final_training(dataset=3, trials=20, hyperparams=hyperparams_3,
#               info='noreg_')


# with regularization
hyperparams_3_reg = dict(
        eta=0.6,
        hidden_sizes=[3],
        alpha=0.9,
        reg_method='l2', reg_lambda=0.01,
        epochs=1000,
        batch_size='batch',  # 'batch',
        activation='sigmoid',
        task='classifier',
        # early_stop='testing',  # 'testing',
        epsilon=5,
        # w_method='DL',
        # w_par=6,
        w_method='uniform',
        w_par=1./17
)

# final_training(dataset=3, trials=10, hyperparams=hyperparams_3_reg,
#               info='withreg_')

'''
# deep
hyperparams_3_deep = dict(
        eta=0.6,
        hidden_sizes=[3, 3],
        alpha=0.9,
        reg_method='l2', reg_lambda=0.01,
        epochs=1000,
        batch_size='batch',  # 'batch',
        activation='sigmoid',
        task='classifier',
        # early_stop='testing',  # 'testing',
        epsilon=5,
        # w_method='DL',
        # w_par=6,
        w_method='uniform',
        w_par=1./17
)

final_training(dataset=3, trials=20, hyperparams=hyperparams_3_deep, info='deep_')
'''
