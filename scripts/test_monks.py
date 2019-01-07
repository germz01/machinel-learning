import numpy as np
import pandas as pd
import validation as val

monks = [['../data/monks/monks-1_train_bin.csv',
          '../data/monks/monks-1_test_bin.csv'],
         ['../data/monks/monks-2_train_bin.csv',
          '../data/monks/monks-2_test_bin.csv'],
         ['../data/monks/monks-3_train_bin.csv',
          '../data/monks/monks-3_test_bin.csv']]

param_ranges = dict()
param_ranges['eta'] = (0.02, 2.0)
param_ranges['alpha'] = 0.001
param_ranges['batch_size'] = (1, 100)
param_ranges['hidden_sizes'] = [(1, 100), (10, 20)]
param_ranges['reg_lambda'] = (0.0, 0.1)
param_ranges['reg_method'] = 'l2'
param_ranges['epochs'] = 200

grid_size = 10
grid = val.HyperGrid(param_ranges, size=grid_size)

for i in np.arange(len(monks)):
    print 'TESTING DATASET MONK {}'.format(i + 1)

    train_set = pd.read_csv(monks[i][0],
                            names=['class'] +
                                  ['x{}'.format(j) for j in range(17)]).values
    test_set = pd.read_csv(monks[i][1],
                           names=['class'] +
                                 ['x{}'.format(j) for j in range(17)]).values

    selection = val.ModelSelectionCV(grid, repetitions=1)
    selection.search(train_set[:, 1:], train_set[:, 0].reshape(-1, 1),
                     save_results=False)

    break
