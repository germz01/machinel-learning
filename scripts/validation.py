from __future__ import division

import nn
import numpy as np

from sklearn.utils.extmath import cartesian
from tqdm import tqdm


class KFoldCrossValidation(object):
    """
    This class represents a wrapper for the implementation of the classic
    k-fold cross validation algorithm, as described in Deep Learning pag.
    120.
    """

    def __init__(self, X, y, nfold=3, **kwargs):
        """
        The class' constructor.

        Parameters
        ----------
        X: numpy.ndarray
            the design matrix

        y: numpy.ndarray
            the target column vector

        nfold: int
            the number of folds to be applied in the algorithm
            (Default value = 3)

        kwargs: dict
            a dictionary which contains the parameters for the neural
            network's initialization

        Returns
        -------
        """
        assert X.shape[0] == y.shape[0]

        self.full_dataset = np.hstack((X, y.reshape(-1, 1)))
        self.folds = list()
        self.results = list()
        self.mean_results = 0.0
        self.record_per_fold = int(self.full_dataset.shape[0] / nfold)
        self.low = 0
        self.high = self.low + self.record_per_fold

        np.random.shuffle(self.full_dataset)
        self.set_folds(nfold)

        self.neural_net = nn.NeuralNetwork(kwargs['hidden_sizes'])

        self.validate(X, y, nfold, eta=kwargs['eta'], alpha=kwargs['alpha'],
                      epochs=kwargs['epochs'], batch_size=kwargs['batch_size'],
                      reg_lambda=kwargs['reg_lambda'],
                      reg_method=kwargs['reg_method'])

    def set_folds(self, nfold):
        """
        This function splits the dataset into nfold folds.

        Parameters
        ----------
        nfold: int
            the number of folds to be applied in the algorithm

        Returns
        -------
        """
        for i in np.arange(nfold):
            self.folds.append(self.full_dataset[self.low:self.high] if i !=
                              nfold - 1 else self.full_dataset[self.low:])

            self.low = self.high
            self.high += self.record_per_fold

    def validate(self, X, y, nfold, **kwargs):
        """
        This function implements the core of the k-fold cross validation
        algorithm. For each fold, the neural network is trained using the
        training set created for that fold, and is tested on the respective
        test set. Finally, the error between the test's target and the
        predicted one is collected.

        Parameters
        ----------
        X. numpy.ndarray
            the design matrix

        y: numpy.ndarray
            the target column vector

        nfold: int
            the number of folds to be applied in the algorithm

        kwargs: dict
            a dictionary which contains the parameters for the neural
            network's initialization

        Returns
        -------
        """
        for i in tqdm(np.arange(nfold),
                      desc='{}-FOLD CROSS VALIDATION PROGRESS'.format(nfold)):
            train_set_full = np.vstack([self.folds[j] for j in np.arange(
                len(self.folds)) if j != i])
            train_set = train_set_full[:, :-1]
            train_target = train_set_full[:, -1].reshape(-1, 1)
            test_set = self.folds[i][:, :-1]
            test_target = self.folds[i][:, -1].reshape(-1, 1)

            self.neural_net.train(train_set, train_target, kwargs['eta'],
                                  kwargs['alpha'], kwargs['epochs'],
                                  kwargs['batch_size'], kwargs['reg_lambda'],
                                  kwargs['reg_method'])

            loss = self.neural_net.predict(test_set, test_target)
            self.results.append(loss)
            # self.neural_net.reset()

        self.mean_result = np.mean(self.results)


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
