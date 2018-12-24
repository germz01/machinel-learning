from __future__ import division

import nn
import numpy as np

from sklearn.utils.extmath import cartesian
from tqdm import tqdm


def kfold_cross_validation(X, y, nfold=3, **kwargs):
    """
    An implementation of the k-fold cross validation algorithm as described in
    Deep Learning, pag 120. The default value for the number of folds is 3,
    and it is possible to pass along all the hyperparameters for the
    neural network's initialization.

    Parameters
    ----------
    X : the design matrix

    y : the target column vector

    nfold : the number of folds to be applied in the algorithm
         (Default value = 3)

    **kwargs : a dictionary which contains the parameters for the neural
               network's initialization

    Returns
    -------
    A list containing the validation scores for each one of the algorithm's
    iterations.
    """
    assert X.shape[0] == y.shape[0]

    full_dataset = np.hstack((X, y.reshape(-1, 1)))
    np.random.shuffle(full_dataset)
    folds = list()
    results = list()

    record_per_fold = int(full_dataset.shape[0] / nfold)
    low = 0
    high = low + record_per_fold

    for i in np.arange(nfold):
        folds.append(full_dataset[low:high] if i != nfold - 1 else
                     full_dataset[low:])

        low = high
        high += record_per_fold

    neural_net = nn.NeuralNetwork(X.shape[1], kwargs['hidden_sizes'], 1)

    for i in tqdm(np.arange(nfold),
                  desc='{}-FOLD CROSS VALIDATION PROGRESS'.format(nfold)):
        train_set = [folds[j] for j in np.arange(len(folds)) if j != i]
        train_set = np.vstack(train_set)
        train_set, train_target = train_set[:, :-1], train_set[:, -1].\
            reshape(-1, 1)
        test_set, test_target = folds[i][:, :-1], folds[i][:, -1].\
            reshape(-1, 1)

        neural_net.train(train_set, train_target, kwargs['eta'],
                         kwargs['alpha'], kwargs['regularizer'],
                         kwargs['epochs'])

        loss = neural_net.predict(test_set, test_target)
        results.append(loss)
        neural_net.reset()

    return results


def grid_search(X, y, random_search=False, **kwargs):
    """
    An implementation of the grid search for the hyperparameters' optimization.
    It is possibile to randomize the whole process by setting the
    random_search variable to True.

    Parameters
    ----------
    X : the design matrix

    y : the target column vector

    random_search : whether or not to proceed with a random hyperparameters's
                    initialization
         (Default value = False)

    **kwargs : the hyperparameters

    Returns
    -------
    A dictionary containing the best validation score among the ones retrieved
    during the searching process. The dictionary contains the best validation
    score and the hyperparameters that were used in order to produce that
    score.
    """
    results = dict()

    if random_search:
        etas = np.random.uniform(0.001, 0.1, 5)
        alphas = np.random.uniform(0, 1, 5)
        regularizers = np.random.uniform(0.001, 0.01, 5)
        epochs = np.random.randint(1, 1000 + 1, 5)

    grid = cartesian([etas, alphas, regularizers, epochs])

    for record in tqdm(grid, desc='GRID SEARCH PROGRESS'):
        val_scores = kfold_cross_validation(X, y, hidden_sizes=[10],
                                            eta=record[0], alpha=record[1],
                                            regularizer=[record[2], 'l2'],
                                            epochs=int(record[3]))
        results[np.mean(val_scores)] = record

    to_ret = {'val_score': min(results.keys()),
              'eta': results[min(results.keys())][0],
              'alpha': results[min(results.keys())][1],
              'lambda': results[min(results.keys())][2],
              'epochs': int(results[min(results.keys())][3])}

    return to_ret
