import numpy as np

from scipy.special import expit

# A_F = {'activation_name': {'f':f(), 'fdev':fdev(), 'range': (a,b) }}

A_F = {
    'identity':
    {
        'f': lambda x: x,
        'fdev': lambda x: 1,
        'range': (np.NINF, np.Inf)
    },
    'sigmoid':
    {
        'f': lambda x: expit(x),
        'fdev': lambda x: expit(x) * (1. - expit(x)),
        'range': (0, 1)
    },
    'tanh':
    {
        'f': lambda x: np.tanh(x),
        'fdev': lambda x: 1 - np.tanh(x)**2,
        'range': (-1, -1)
    },
    'relu':
    {
        'f': lambda x: 0 if x < 0 else x,
        'fdev': lambda x: 0 if x < 0 else 1,
        'range': (0, np.Inf)
    }
}
