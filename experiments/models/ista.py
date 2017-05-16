import sys
import numpy as np

from ._cost import LossL2, ConvL2_z
from .utils import cost_network, cost_conv_network, soft_thresholding


def _log(progress, cost, verbose=0):
    if verbose > 0:
        sys.stdout.write(f"\rProgress: {progress:7.2%} - {cost:15.6f}")
        sys.stdout.flush()


def ista(X, D, lmbd, max_iter, z0=None, verbose=0):
    """ISTA for X and D

    Parameters
    ----------
    X (array-like - (N, p)): data to compute the sparse code.
        There is N samples in a p dimensional space.
    D (array-like - (K, p)): dictionary to compute the sparse code.
        There is K atoms constituted by p-dimensional points.
    lmbd (float): regularization parameter for the sparse coding
    max_iter (int): number of iteration for ISTA

    Return
    ------
    zk (array-like - (N, K)): code for the given data point
    cost (list): cost at each iteration of the algorithm

    """
    N, K = X.shape[0], D.shape[0]
    L = np.linalg.norm(D, ord=2)**2

    if z0 is None:
        zk = np.zeros(shape=(N, K))
    else:
        zk = np.copy(z0)

    _cost_model = cost_network((X.shape[1:], zk.shape[1:]), D, lmbd, False)
    _cost = lambda zk: _cost_model([X, zk]).mean()

    c = _cost(zk)
    cost = [c]
    _log(0, c, verbose)
    for i in range(max_iter):
        grad = (zk.dot(D) - X).dot(D.T)
        zk -= grad / L
        zk = soft_thresholding(zk, lmbd / L)
        c = _cost(zk)
        cost += [c]
        _log((i + 1) / max_iter, c, verbose)

    if verbose > 0:
        print()

    return zk, cost


def fista(X, D, lmbd, max_iter, z0=None, verbose=0):
    """Fast ISTA for X and D

    Parameters
    ----------
    X (array-like - (N, p)): data to compute the sparse code.
        There is N samples in a p dimensional space.
    D (array-like - (K, p)): dictionary to compute the sparse code.
        There is K atoms constituted by p-dimensional points.
    lmbd (float): regularization parameter for the sparse coding
    max_iter (int): number of iteration for FISTA

    Return
    ------
    zk (array-like - (N, K)): code for the given signals
    cost (list): cost at each iteration of the algorithm

    """
    f_cost = LossL2(X, D)

    if z0 is None:
        zk = f_cost.get_z0()
    else:
        zk = np.copy(z0)
    momentum = 1
    y = zk

    _cost_model = cost_network((X.shape[1:], zk.shape[1:]), D, lmbd, False)
    _cost = lambda zk: _cost_model([X, zk]).mean()

    c = _cost(zk)
    cost = [c]
    _log(0, c, verbose)
    for i in range(max_iter):

        z_old, momentum_1 = zk, momentum

        zk = y - f_cost.grad(y) / f_cost.L
        zk = soft_thresholding(zk, lmbd / f_cost.L)
        momentum = (1 + np.sqrt(1 + 4 * momentum * momentum)) / 2
        y = zk + (momentum_1 - 1) / momentum * (zk - z_old)

        c = _cost(zk)
        if c >= cost[-1]:
            # Restart the momentum if cost increase
            zk = z_old - f_cost.grad(z_old) / f_cost.L
            y = zk = soft_thresholding(zk, lmbd / f_cost.L)
            c = _cost(zk)
        cost += [c]
        _log((i + 1) / max_iter, c, verbose)

    if verbose > 0:
        print()

    return zk, cost


def ista_conv(X, D, lmbd, max_iter, z0=None, verbose=0):
    """Convolutional ISTA for X and D

    Parameters
    ----------
    X (array-like -(N, p, w, h)): signal to compute the sparse code.
        There is N samples with p dimensional 2d signals of size wxh.
    D (array-like - (K, p, kw, kh)): dictionary to compute the sparse code.
        There is K atoms constituted by 2d p-dimensional signals of size kwxkh.
    lmbd (float): regularization parameter for the sparse coding
    max_iter (int): number of iteration for FISTA

    Return
    ------
    zk (array-like - (N, K, w-wk+1, h-hk+1)): code for the given signals
    cost (list): cost at each iteration of the algorithm

    """

    # Initiate the algorithm
    f_cost = ConvL2_z(X, D)
    if z0 is None:
        zk = f_cost.get_z0()
    else:
        zk = np.copy(z0)
    L = f_cost.L

    # Define _cost function using keras to get conscistent results
    # def _cost(z):
    #     return f_cost(z) + lmbd * abs(z).mean(axis=0).sum()
    Dk = np.transpose(D, (2, 3, 0, 1))[::-1, ::-1]
    if zk.ndim == 5:
        zk_shape = (zk.shape[1],) + zk.shape[3:]
        _cost_model = cost_conv_network((X.shape[1:], zk_shape), Dk, lmbd)
        _cost = lambda zk: _cost_model([X, zk[:, :, 0]]).mean()
    else:
        _cost_model = cost_conv_network((X.shape[1:], zk.shape[1:]), Dk, lmbd)
        _cost = lambda zk: _cost_model([X, zk]).mean()

    c = _cost(zk)
    cost = [c]
    _log(0, c, verbose)
    for i in range(max_iter):
        zk -= f_cost.grad(zk) / L
        zk = soft_thresholding(zk, lmbd / L)
        c = _cost(zk)
        cost += [c]
        _log((i + 1) / max_iter, c, verbose)
    if verbose > 0:
        print()

    return zk, cost


def fista_conv(X, D, lmbd, max_iter, z0=None, verbose=0):
    """Convolutional fast ISTA for X and D

    Parameters
    ----------
    X (array-like - (N, p, w, h)): signal to compute the sparse code.
        There is N samples with p dimensional 2d signals of size wxh.
    D (array-like - (K, p, kw, kh)): dictionary to compute the sparse code.
        There is K atoms constituted by 2d p-dimensional signals of size kwxkh.
    lmbd (float): regularization parameter for the sparse coding
    max_iter (int): number of iteration for FISTA

    Return
    ------
    zk (array-like - (N, K, w-wk+1, h-hk+1)): code for the given signals
    cost (list): cost at each iteration of the algorithm

    """

    # Initiate the algorithm
    f_cost = ConvL2_z(X, D)
    if z0 is None:
        zk = f_cost.get_z0()
    else:
        zk = np.copy(z0)
    L = f_cost.L

    # Define _cost function using keras to get conscistent results
    # def _cost(z):
    #     return f_cost(z) + lmbd * abs(z).mean(axis=0).sum()
    Dk = np.transpose(D, (2, 3, 0, 1))[::-1, ::-1]
    if zk.ndim == 5:
        zk_shape = (zk.shape[1],) + zk.shape[3:]
        _cost_model = cost_conv_network((X.shape[1:], zk_shape), Dk, lmbd)
        _cost = lambda zk: _cost_model([X, zk[:, :, 0]]).mean()
    else:
        _cost_model = cost_conv_network((X.shape[1:], zk.shape[1:]), Dk, lmbd)
        _cost = lambda zk: _cost_model([X, zk]).mean()

    momentum = 1
    y = zk
    c = _cost(zk)
    cost = [c]
    _log(0, c, verbose)
    for i in range(max_iter):

        z_old, momentum_1 = zk, momentum

        grad = f_cost.grad(y)
        zk = y - grad / L
        zk = soft_thresholding(zk, lmbd / L)
        momentum = (1 + np.sqrt(1 + 4 * momentum * momentum)) / 2
        y = zk + (momentum_1 - 1) / momentum * (zk - z_old)

        c = _cost(zk)
        if c >= cost[-1]:
            # Restart the momentum if cost increase
            zk = z_old - f_cost.grad(z_old) / L
            y = zk = soft_thresholding(zk, lmbd / L)
            c = _cost(zk)

        cost += [c]
        _log((i + 1) / max_iter, c, verbose)
    if verbose > 0:
        print()

    return zk, cost
