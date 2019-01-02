import nn
import numpy as np
import validation as val

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
    neural_net = nn.NeuralNetwork(X, y)

    cross_val = val.KFoldCrossValidation(X, y, neural_net,
                                         eta=eta, alpha=alpha, epochs=500,
                                         batch_size=10, reg_lambda=0.01,
                                         reg_method='l2')
    tqdm.write('VALIDATION ERRORS {}'.format(cross_val.results))
    tqdm.write('MEAN VALIDATION ERROR: {}'.format(cross_val.mean_result))
    tqdm.write('VARIANCE FOR VALIDATION ERROR {}'.format(cross_val.std_result))

if raw_input('TESTING GRID SEARCH?[Y/N] ') == 'Y':
    par_ranges = dict()
    par_ranges['eta'] = (0.1, 0.9)
    par_ranges['alpha'] = (0.1, 0.9)
    par_ranges['reg_lambda'] = (0.001, 0.01)
    par_ranges['batch_size'] = (1, 100)
    par_ranges['epochs'] = (10, 1000)

    grid_search = val.GridSearch(X, y, random_search=True,
                                 par_ranges=par_ranges)
    print '\n\nBEST RESULT {} FROM RECORD: {}'.\
        format(grid_search.best_result['error'],
               grid_search.best_result['parameters'])
