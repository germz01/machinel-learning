import nn as NN
import numpy as np
import matplotlib.pyplot as plt
import imp
import utils as u
import validation as val

X = np.concatenate((np.random.normal(2., 1., (50, 2)),
                    np.random.normal(5., 1., (50, 2))),
                   axis=0)
y = np.hstack((np.ones(50), np.zeros(50))).reshape(100, 1)


imp.reload(NN)
imp.reload(u)

eta = .1
alpha = .9

if raw_input('PLAIN EXECUTION WITHOUT TESTING?[Y/N] ') == 'Y':
    nn = NN.NeuralNetwork(X.shape[1], [10], 1)
    nn.train(X, y, eta, alpha, [0.01, 'l2'], 1000)

    if raw_input('PLOT LEARNING CURVE?[Y/N] ') == 'Y':
        epochs = 500
        # nota: loss online e epochs sono su range diversi
        # da sistemare successivamente a batch/minibatch/online implementation
        fig, ax = plt.subplots(figsize=(5, 7))
        plt.plot(range(len(nn.loss_online[:epochs])), nn.loss_online[:epochs])
        plt.plot(range(len(nn.loss_epochs[:epochs])), nn.loss_epochs[:epochs])

        plt.savefig('../images/temp_loss.pdf')
        plt.close()

if raw_input('TESTING K-FOLD CROSS VALIDATION?[Y/N] ') == 'Y':
    cross_val = val.KFoldCrossValidation(X, y, nfold=5, hidden_sizes=[10],
                                         eta=eta, alpha=alpha, epochs=500,
                                         batch_size=10, reg_lambda=0.01,
                                         reg_method='l2')
    print 'VALIDATION ERRORS {}'.format(cross_val.results)
    print 'MEAN VALIDATION ERROR: {}'.format(cross_val.mean_result)

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
