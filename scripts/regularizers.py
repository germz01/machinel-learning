import numpy as np


def l1(w, lamb):
    """
    This functions implements the L1 regularization for the weights' decay
    as described in Deep Learning, pag. 228.

    Parameters
    ----------
    w : a matrix of weights

    lamb : the regularization constant


    Returns
    -------
    The L1 regularization factor
    """
    return lamb * np.sign(w)


def l2(w, lamb):
    """
    This functions implements the L2 regularization for the weights' decay
    as described in Deep Learning, pag. 224.
    Parameters
    ----------
    w : a matrix of weights

    lamb : the regularization constant


    Returns
    -------
    The L2 regularization factor
    """
    return lamb * w
