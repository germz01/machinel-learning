import numpy as np
import pandas as pd

import nn as NN
import utils as u
import validation as valid
import metrics

import time
import json
import os.path
import csv
import imp

from pprint import pprint

###########################################################
# EXPERIMENTAL SETUP

dataset = 1

grid_size = 100

nfolds = 3
ntrials = 3

param_ranges = {
    'eta': (0.0001, 2.0),
    'hidden_sizes': [(2, 100)],
    'alpha': (0.0, 0.9),
    'reg_method': 'l2', 'reg_lambda': 0.0,
    'epochs': 1000,
    'batch_size': 'batch',
    'activation': 'sigmoid',
    'task': 'classifier',
    # 'early_stop': None,  # 'testing',
    'epsilon': 5,
    'w_method': 'uniform',
    'w_par': 1.0}

info = "Informazioni/appunti/scopo riguardo l'esperimento in corso"

info = "Initial search, without regularization, using sigmoid"

experiment_params = {
    'dataset': dataset,
    'nfolds': nfolds,
    'grid_size': grid_size,
    'info': info,
    'param_ranges': param_ranges
}

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

design_set = datasets['monks-{}_train'.format(dataset)]
test_set = datasets['monks-{}_test'.format(dataset)]

y_design, X_design = np.hsplit(design_set, [1])

# simmetrized X_design:
X_design = X_design*2-1


###########################################################
# babysitting for choosing initial param ranges
# and plot_learning_curve


np.random.shuffle(design_set)

# splitting training/validation
split_percentage = 0.75
split = int(design_set.shape[0]*split_percentage)

training_set = design_set[:split, :]
validation_set = design_set[split:, :]

y_training, X_training = np.hsplit(training_set, [1])
y_validation, X_validation = np.hsplit(validation_set, [1])


imp.reload(u)
imp.reload(NN)

nn = NN.NeuralNetwork(X_training, y_training,
                      eta=1.1,
                      hidden_sizes=[100],
                      alpha=0.7,
                      reg_method='l2', reg_lambda=0.0,
                      epochs=1000,
                      batch_size=X_training.shape[0],
                      activation='sigmoid',
                      task='classifier',
                      early_stop='testing',
                      epsilon=5,
                      w_method='uniform',
                      w_par=0.7)
nn.train(X_training, y_training, X_validation, y_validation)
u.plot_learning_curve(nn, fname='../images/monks_learning_curve.pdf')

nn.W


y_pred = nn.predict(X_validation)
y_pred = np.apply_along_axis(lambda x: 0 if x < .5 else 1, 1,
                             y_pred).reshape(-1, 1)


imp.reload(valid)

bca = metrics.BinaryClassifierAssessment(y_pred, y_validation,
                                         printing=True)


bca.accuracy
bca.precision

w_par = 160
W = nn.set_weights(w_par)
W
pd.DataFrame(W[0]).describe()


###########################################################
# EXPERIMENT GRID SEARCH

# controllo nomi files
fpath = '../data/monks/results/'

check_files = True
experiment = 1

while(check_files):
    fname_results = 'monks_{}_experiment_{}_results.json.gz'.format(
        dataset, experiment)
    fname_params = 'monks_{}_experiment_{}_parameters.json'.format(
        dataset, experiment)

    fres = fpath+fname_results
    fpar = fpath+fname_params

    if os.path.isfile(fres) or os.path.isfile(fpar):
        experiment += 1
    else:
        check_files = False

print '--------------------'
print 'STARTING EXPERIMENT'
print 'saving results in:'
print fres
print fpar
print '--------------------'
# save experiment setup
with open(fpar, 'w') as f:
    json.dump(experiment_params, f, indent=4)

float(np.sum(y_training))/len(y_training)
float(np.sum(y_validation))/len(y_validation)

###########################################################
# starting search

grid = valid.HyperGrid(param_ranges, grid_size, random=True)
selection = valid.ModelSelectionCV(grid, fname=fres)

start = time.time()
selection.search(X_design, y_design, nfolds=nfolds, ntrials=ntrials)
end = time.time()

print end-start

###########################################################
