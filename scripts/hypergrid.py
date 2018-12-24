from itertools import product
import random
import numpy as np


class HyperGrid():
    """
    HyperGrid class to instanciate a grid iterator object

    Attributes
    ----------
    grid_f: dict
        A dictionary containing function to generate parameters arrays
        using the input ranges

    """
    grid_f = {
        'linspace':
        {
            int: lambda start, end, num:
            np.linspace(start, end, num, dtype=int),
            float: lambda start, end, num:
            np.linspace(start, end, num, dtype=float)
        },
        'random':
        {
            int: lambda start, end, num:
            np.random.randint(start, end, num, dtype=int),
            float: lambda start, end, num:
            np.random.uniform(start, end, num)
        }
    }

    def __init__(self, param_ranges, size, method):
        """
        Initialize

        Parameters
        ----------
        param_ranges : dict
            dictionary containing hyperparameters ranges

        size : int
            size of each hyperparameter array

        method : str
            'linspace' to generate a uniform grid space
            'random' to generate a random grid

        Returns
        -------

        """

        self.param_ranges = param_ranges
        self.size = size
        self.method = method
        self.params = param_ranges.keys()
        self.params_index = {par: self.params.index(par)
                             for par in self.params}
        # set seed for the hyperGrid object
        self.seed = random.randint(0, 2**32)

        print ('GENERATING AN HYPERPARAMETER GRID OF LENGTH {}'
               .format(self.get_grid_size()))

    def get_grid(self):
        """
        Produces the grid iterator

        Returns
        -------
        grid_iter: itertools.product
            a grid iterator
        """
        np.random.seed(seed=self.seed)
        par_vectors = dict()
        for par, interval in self.param_ranges.items():
            par_vectors[par] = self.grid_f[self.method][type(interval[0])](
                interval[0],
                interval[1],
                self.size
            )
            # return par_vectors
        grid_iter = product(*list(par_vectors.values()))
        return grid_iter

    def get_grid_list(self):
        """ get the grid as a list """
        grid_list = list(self.get_grid())

        return grid_list

    def get_grid_dict(self):
        """ get the grid as a dict """
        grid_iter = self.get_grid()
        grid_dict = {par: [] for par in self.params}
        for par_values in grid_iter:
            for par in self.params:
                grid_dict[par].append(
                    par_values[self.params_index[par]]
                )
        return grid_dict

    def get_grid_size(self):
        """ get the grid size """
        return self.size**len(self.params)

    # ##########################################################


if __name__ == "__main__":

    from pprint import pprint

    par_ranges = dict()
    par_ranges['eta'] = (5.0, 10.0)
    par_ranges['alpha'] = (0.01, 0.9)
    par_ranges['batch_size'] = (1, 100)

    # problema: in caso di metodo random non e possibile rigenerare la
    # stessa griglia, pero' si: basta usare un seed fisso alla creazione
    # della griglia stessa da non cambiare piu

    grid = HyperGrid(par_ranges, 2, 'random')

    grid.get_grid_size()

    grid_iter = grid.get_grid()
    for val in grid_iter:
        print val
    print '-- grid as dict --'
    pprint(grid.get_grid_dict())
    print '-- grid as list --'
    pprint(grid.get_grid_list())
