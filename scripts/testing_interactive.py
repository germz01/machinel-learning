import nn
import numpy as np
import validation as val

from pprint import pprint
from tqdm import tqdm

X = np.concatenate((np.random.normal(2., 1., (50, 2)),
                    np.random.normal(5., 1., (50, 2))),
                   axis=0)
y = np.hstack((np.ones(50), np.zeros(50))).reshape(100, 1)

eta = .1
alpha = .9

if raw_input('PLAIN EXECUTION WITHOUT TESTING?[Y/N] ') == 'Y':
    neural_net = nn.NeuralNetwork(X, y)
    neural_net.train(X, y, eta, alpha=alpha, epochs=500, batch_size=10,
                     reg_lambda=0.001, reg_method='l2')

    if raw_input('PLOT LEARNING CURVE?[Y/N] ') == 'Y':
        pass

if raw_input('TESTING K-FOLD CROSS VALIDATION?[Y/N] ') == 'Y':
    neural_net = nn.NeuralNetwork(X, y, eta=eta, alpha=alpha, epochs=500,
                                  batch_size=10, reg_lambda=0.01,
                                  reg_method='l2')

    cross_val = val.KFoldCrossValidation(X, y, neural_net, nfolds=5,
                                         eta=eta, alpha=alpha, epochs=500,
                                         batch_size=10, reg_lambda=0.01,
                                         reg_method='l2')
    tqdm.write('AGGREGATED RESULTS: \n')
    pprint(cross_val.aggregated_results)

if raw_input('TESTING GRID SEARCH?[Y/N] ') == 'Y':
    param_ranges = dict()
    param_ranges['eta'] = (0.02, 2.0)
    param_ranges['alpha'] = 0.001
    param_ranges['batch_size'] = (1, 100)
    param_ranges['hidden_sizes'] = [(1, 100), (10, 20)]
    param_ranges['reg_lambda'] = (0.0, 0.1)
    param_ranges['reg_method'] = 'l2'
    param_ranges['epochs'] = 200

    grid_size = 10
    grid = val.HyperRandomGrid(param_ranges, N=grid_size)

    grid = [v for v in grid]

    selection = val.ModelSelectionCV(grid)
    selection.search(X, y)
    results = selection.load_results()

    best_model = selection.select_best_model(X, y)

    # print best_model

    # grid_search = val.GridSearch(X, y, random_search=True,
    #                              par_ranges=par_ranges)
    # print '\n\nBEST RESULT {} FROM RECORD: {}'.\
    #     format(grid_search.best_result['error'],
    #            grid_search.best_result['parameters'])
