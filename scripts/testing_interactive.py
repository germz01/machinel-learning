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
    results = val.kfold_cross_validation(X, y, nfold=5, hidden_sizes=[10],
                                         eta=eta, alpha=alpha,
                                         regularizer=[0.01, 'l2'],
                                         epochs=1000)
    print '\nMEAN VALIDATION ERROR: {}'.format(np.mean(results))

if raw_input('TESTING GRID SEARCH?[Y/N] ') == 'Y':
    best_val_score = val.grid_search(X, y, random_search=True)

    print '\n'
    print best_val_score

