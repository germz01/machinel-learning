import nn as NN
import numpy as np
import matplotlib.pyplot as plt
import imp
import utils as u
import pandas as pd
from pprint import pprint
import validation as val
import time


from sklearn.preprocessing import StandardScaler


df = pd.read_csv('../data/energydata_complete.csv')
''' 
dataset da
https://archive.ics.uci.edu/ml/datasets/energy+efficiency 
'''

df = df.drop(['date', 'rv1', 'rv2'], axis = 1)

df.columns

dataset = df.values

scaler = StandardScaler()
scaler.fit(dataset)
dataset = scaler.transform(dataset)


np.random.seed(10)

np.random.shuffle(dataset)

dataset.shape

dataset = dataset[:1000, :]

dataset[1,1]

y = dataset[:, 0].reshape(-1,1)
X = dataset[:, 1:]

y.shape
X.shape

imp.reload(NN)
imp.reload(u)

dataset = np.hstack((X, y))

split = int(X.shape[0] * 0.7)

np.random.shuffle(dataset)
train = dataset[:split, :]
val = dataset[split:, :]

train.shape
val.shape

X_train, y_train = np.hsplit(train, [X.shape[1]])
X_val, y_val = np.hsplit(val, [X.shape[1]])

nn = NN.NeuralNetwork(X_train, y_train,
                      eta=0.07,
                      hidden_sizes = [15], 
                      alpha=0.5,
                      reg_method='l2', reg_lambda=0.,
                      epochs=1000,
                      batch_size=X_train.shape[0],
                      activation = 'relu',
                      task = 'regression',
                      w_par=6)
nn.train(X_train, y_train, X_val, y_val)
u.plot_learning_curve(nn)

nn.error_per_epochs[-1]


nn.error_per_epochs_mse[-1]
nn.error_per_epochs_val[-1]


###########################################################

# Testing Cross Validation

# defining grid
param_ranges = dict()
param_ranges['eta'] = (0.001, 2.0)
param_ranges['alpha'] = 0.001
param_ranges['batch_size'] = (1, 100)
param_ranges['hidden_sizes'] = [(1, 100), (10, 20)]
param_ranges['reg_lambda'] = (0.0, 0.1)
param_ranges['reg_method'] = 'l2'
param_ranges['epochs'] = 200

imp.reload(val)

# uniform grid
grid_size = 3
grid = val.HyperGrid(param_ranges, size=grid_size, random=False)

i=0
for hyperparam in grid:
    pprint(hyperparam)
    i+=1

# random grid
grid_size = 10
grid = val.HyperGrid(param_ranges, size=grid_size, random=True)
len(grid)



selection = val.ModelSelectionCV(grid=grid, repetitions=2)

start = time.time()
selection.search(X, y, nfolds=3)
end = time.time()

print end-start
results = selection.load_results()

pprint(selection.select_best_hyperparams(top=7))

best_model = selection.select_best_model(X, y)
