import numpy as np
import utils as u
import pandas as pd
import nn as NN


class Holdout():
    """ Validation Holdout method """
    def __init__(self, X, y, split_perc=[0.5, 0.25, 0.25]):
        """
        Initialization for the Holdout class

        Parameters
        ----------
        X : numpy.ndarray
        y : numpy.ndarray
        split_perc : list
            split percentages

        Returns
        -------

        """
        df = np.hstack((X, y))
        np.random.shuffle(df)

        p = df.shape[0]
        tr_perc = split_perc[0]
        va_perc = split_perc[1]
        # ts_perc = split_perc[2]

        split_train = int(tr_perc*p)
        split_design = int((tr_perc+va_perc)*p)

        design_set = df[:split_design, :]
        train_set = df[:split_train, :]
        validation_set = df[split_train:split_design, :]
        test_set = df[split_design:, :]

        self.X_design, self.y_design = np.hsplit(design_set, [X.shape[1]])
        self.X_train, self.y_train = np.hsplit(train_set, [X.shape[1]])
        self.X_va, self.y_va = np.hsplit(validation_set, [X.shape[1]])
        self.X_test, self.y_test = np.hsplit(test_set, [X.shape[1]])

    def model_selection(self, grid, plot=False, fpath='../images/'):
        """
        Holdout model selection

        Parameters
        ----------
        grid : instance of HyperRandomGrid class
            hyperparameter grid
        plot : bool
            if plot=True plots the learning curve for each grid parameter

        fpath : str
            path for images storing
        Returns
        -------
        neural network object
        """

        self.fpath = fpath
        params = []
        errors_va = []
        for i, pars in enumerate(grid):
            # TODO: handle multiple layers
            nn = NN.NeuralNetwork(hidden_sizes=[pars.pop('hidden_sizes')])
            nn.train(self.X_train, self.y_train, epochs=500, **pars)
            print('trained')
            params.append(nn.get_params())
            # assess on validation set
            errors_va.append(
                nn.predict(self.X_va, self.y_va)/(self.X_va.shape[0])
            )
            if plot is True:
                u.plot_error(nn, fname=fpath
                             + 'learning_curve_{}.png'.format(i))

        # choosing the best hyperparameters
        self.best_index = np.argmin(errors_va)
        best_hyperparams = params[self.best_index]

        # retraining on design set
        nn_retrained = NN.NeuralNetwork(hidden_sizes=best_hyperparams
                                        .pop('hidden_sizes'))
        nn_retrained.train(self.X_design, self.y_design, **best_hyperparams)

        df_pars = pd.DataFrame(list(grid))
        df_pars['error'] = errors_va

        self.best_hyperparams = best_hyperparams
        self.df_pars = df_pars
        self.model = nn_retrained

        return self.model
