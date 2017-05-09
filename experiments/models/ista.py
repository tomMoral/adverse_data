import sys
import numpy as np
from scipy import fftpack
from numpy.fft import rfftn as fft, irfftn as ifft

from ._cost import ConvL2_z


def soft_thresholding(x, mu):
    """Compute the soft thresholding"""
    return np.sign(x) * np.maximum(abs(x) - mu, 0)


def _log(progress, cost, verbose=0):
    if verbose > 0:
        sys.stdout.write(f"\rProgress: {progress:7.2%} - {cost:15.6f}")
        sys.stdout.flush()


def sc_cost(x, D, zk, lmbd):
    err = x - zk.dot(D)
    l2 = np.sum(err * err, axis=1).mean()
    l1 = np.sum(abs(zk), axis=1).mean()
    return 1 / 2 * l2 + lmbd * l1


def ista(X, max_iter, D, lmbd):
    N, K = X.shape[0], D.shape[0]
    L = np.linalg.norm(D, ord=2)**2

    zk = np.zeros(shape=(N, K))
    cost = []
    for i in range(max_iter):
        grad = (zk.dot(D) - X).dot(D.T)
        zk -= grad / L
        zk = soft_thresholding(zk, lmbd / L)
        c = sc_cost(X, D, zk, lmbd)
        cost += [c]
        sys.stdout.write(f"\rProgress: {i/max_iter:7.2%} - {c:.4f}")
        sys.stdout.flush()
    print(f"\rProgress: {max_iter/max_iter:7.2%} - {c:.4f}")

    return zk, cost


def fista(X, max_iter, D, lmbd):
    N, K = X.shape[0], D.shape[0]
    L = np.linalg.norm(D, ord=2)**2

    zk = np.zeros(shape=(N, K))
    momentum = 1
    y = zk

    cost = []
    for i in range(max_iter):

        z_old, momentum_1 = zk, momentum

        grad = (y.dot(D) - X).dot(D.T)
        zk = y - grad / L
        zk = soft_thresholding(zk, lmbd / L)
        momentum = (1 + np.sqrt(1 + 4 * momentum * momentum)) / 2
        y = zk + (momentum_1 - 1) / momentum * (zk - z_old)

        c = sc_cost(X, D, zk, lmbd)
        cost += [c]
        sys.stdout.write(f"\rProgress: {i/max_iter:7.2%} - {c:.4f}")
        sys.stdout.flush()
    print(f"\rProgress: {max_iter/max_iter:7.2%} - {c:.4f}")

    return zk, cost


def ista_conv(X, D, lmbd, max_iter, z0=None, verbose=0):
    """Convolutional ISTA for X and D

    Parameters
    ----------
    X (array-like, N, p, w, h): signal to compute the sparse code.
        There is N samples with p dimensional 2d signals of size wxh.
    D (array-like, K, p, kw, kh): dictionary to compute the sparse code.
        There is K atoms constituted by 2d p-dimensional signals of size kwxkh.
    lmbd (float): regularization parameter for the sparse coding
    max_iter (int): number of iteration for FISTA

    Return
    ------
    zk (array-like, N, K, w-wk+1, h-hk+1): code for the given signals
    cost: history of the cost evaluation after each iteration.

    """
    f_cost = ConvL2_z(X, D)

    def _cost(z):
        return f_cost(z) + lmbd * abs(z).mean(axis=0).sum()

    # Initiate the algorithm
    if z0 is None:
        zk = f_cost.get_z0()
    else:
        zk = np.copy(z0)
    L = f_cost.L

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
    X (array-like, N, p, w, h): signal to compute the sparse code.
        There is N samples with p dimensional 2d signals of size wxh.
    D (array-like, K, p, kw, kh): dictionary to compute the sparse code.
        There is K atoms constituted by 2d p-dimensional signals of size kwxkh.
    lmbd (float): regularization parameter for the sparse coding
    max_iter (int): number of iteration for FISTA

    Return
    ------
    zk (array-like, N, K, w-wk+1, h-hk+1): code for the given signals
    cost: history of the cost evaluation after each iteration.

    """
    f_cost = ConvL2_z(X, D)

    def _cost(z):
        return f_cost(z) + lmbd * abs(z).mean(axis=0).sum()

    # Initiate the algorithm
    if z0 is None:
        zk = f_cost.get_z0()
    else:
        zk = np.copy(z0)
    L = f_cost.L

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
            assert f_cost(zk) <= f_cost(z_old)
            y = zk = soft_thresholding(zk, lmbd / L)
            c = _cost(zk)

        cost += [c]
        _log((i + 1) / max_iter, c, verbose)
    if verbose > 0:
        print()

    return zk, cost
