from itertools import product
import random
import numpy as np


class HyperRandomGrid():
    """ HyperRandomGrid """
    def __init__(self, param_ranges, N, seed=None):
        """
        HyperRandomGrid instanciates an iterator
        which produce random parameters, in the given ranges.

        The grid iterator is reset after each use,
        allowing immediate reuse of the same grid.

        Parameters
        ----------
        param_ranges : dict
        dictionary containing ranges interval for each parameter.

        N: int
        size of the grid.

        seed: int
        random seed initialization.

        Returns
        -------

        """

        self.N = N
        self.n = 0
        if type(param_ranges) is not dict:
            raise TypeError("Insert a dictionary of parameters ranges")
        self.param_ranges = param_ranges
        self.types = self.get_types()

        if seed is not None:
            # seed inizialization
            self.seed = seed
            random.seed(self.seed)
        else:
            # random initialization
            random.seed()
            self.seed = random.randint(0, 2**32)
            random.seed(self.seed)

    def get_types(self):
        """
        Get the type of each parameter

        Parameters
        ----------

        Returns
        -------
        types : dict
        dictionary containing each parameter type
        """

        types = dict()
        for par, interval in self.param_ranges.items():
            if type(interval[0]) is int and type(interval[1] is int):
                types[par] = int
            elif type(interval[0]) is float and type(interval[1] is float):
                types[par] = float
            else:
                raise TypeError('Check interval type')
        return types

    def __iter__(self):
        return self

    def next(self):
        """
        Iterator next method,
        returns the next grid record

        Parameters
        ----------
        Returns
        -------
        x_grid : dict
        Randomized parameter dictionary
        """
        if self.n == 0:
            random.seed(self.seed)

        x_grid = dict()
        for par, interval in self.param_ranges.items():
            if self.types[par] is int:
                x_grid[par] = random.randint(interval[0], interval[1])
            else:
                x_grid[par] = random.uniform(interval[0], interval[1])

        self.n += 1
        if self.n == self.N+1:
            self.n = 0
            # set random seed at exit
            random.seed()
            raise StopIteration
        else:
            return x_grid

    def reset_grid(self):
        """Reset the grid, to use again the iterator """
        random.seed(self.seed)
        self.n = 0

    def get_par_index(self, index):
        self.reset_grid()
        for i in range(index+1):
            params = self.next()
        return params


class HyperGrid():
    """
    HyperGrid class  instanciates a grid iterator object

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

    def __init__(self, param_ranges, size, method, ):
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
        self.grid = self.get_grid()

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

    def __iter__(self):
        """
        iterator method
        """
        return self

    def next(self):
        """
        Return next grid parameters as a dictionary
        """
        out = self.grid.next()

        d = dict()
        for i, par in enumerate(self.params):
            d[par] = out[i]

        return d

    def reset_grid(self):
        """
        Re-create the grid iterator, maintaining the same
        parameters
        """
        self.grid = self.get_grid()

    # ##########################################################


if __name__ == "__main__":

    from pprint import pprint

    par_ranges = dict()
    par_ranges['eta'] = (5.0, 10.0)
    par_ranges['alpha'] = (0.01, 0.9)
    par_ranges['batch_size'] = (1, 100)

    grid = HyperRandomGrid(par_ranges, 3)
    print('get grid-list from grid iterator ')
    grid1 = [x for x in grid]
    pprint(grid1)
    print('the grid iterator is re-usable each time')
    grid2 = [x for x in grid]
    pprint(grid2)
