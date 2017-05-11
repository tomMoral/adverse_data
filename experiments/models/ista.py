import sys
import numpy as np
from scipy.signal import convolve2d

# Import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, Add, Lambda

from ._cost import ConvL2_z
from .utils import get_soft_thresholding, get_cost_lasso_layer, cost_network


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


def ista(X, D, lmbd, max_iter, z0=None, verbose=0):
    N, K = X.shape[0], D.shape[0]
    L = np.linalg.norm(D, ord=2)**2

    if z0 is None:
        zk = np.zeros(shape=(N, K))
    else:
        zk = np.copy(z0)

    def _cost(z):
        return sc_cost(X, D, z, lmbd)

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
    N, K = X.shape[0], D.shape[0]
    L = np.linalg.norm(D, ord=2)**2

    if z0 is None:
        zk = np.zeros(shape=(N, K))
    else:
        zk = np.copy(z0)
    momentum = 1
    y = zk

    def _cost(z):
        return sc_cost(X, D, z, lmbd)

    c = _cost(zk)
    cost = [c]
    _log(0, c, verbose)
    for i in range(max_iter):

        z_old, momentum_1 = zk, momentum

        grad = (y.dot(D) - X).dot(D.T)
        zk = y - grad / L
        zk = soft_thresholding(zk, lmbd / L)
        momentum = (1 + np.sqrt(1 + 4 * momentum * momentum)) / 2
        y = zk + (momentum_1 - 1) / momentum * (zk - z_old)

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
        _cost_model = cost_network((X.shape[1:], zk_shape), Dk, lmbd)
        _cost = lambda zk: _cost_model([X, zk[:, :, 0]]).mean()
    else:
        _cost_model = cost_network((X.shape[1:], zk.shape[1:]), Dk, lmbd)
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
    _cost_model = cost_network((X.shape[1:], zk.shape[1:]), Dk, lmbd)
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
            assert f_cost(zk) <= f_cost(z_old)
            y = zk = soft_thresholding(zk, lmbd / L)
            c = _cost(zk)

        cost += [c]
        _log((i + 1) / max_iter, c, verbose)
    if verbose > 0:
        print()

    return zk, cost


def conv_ista_network(X, D, lmbd, max_iter, z0=None, verbose=0):
    """Convolutional ISTA for X and D implemented using keras network

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
    cost (list): cost at each iteration of the algorithm

    """
    input_dim = X.shape[1:]
    d, c = D.shape[:2]
    Wx_size = w, h = D.shape[-2:]

    assert c == X.shape[1], "mismatched dimension between input and dictionary"

    # Compute constants

    f_cost = ConvL2_z(np.zeros((10,) + input_dim), D)
    S = np.array([[[convolve2d(d1, d2, mode="full")
                   for d1, d2 in zip(dk, dl)]
                  for dk in D] for dl in D[:, :, ::-1, ::-1]]).mean(axis=2)
    Wz_size = S.shape[-2:]
    Wz = np.zeros(S.shape, dtype=D.dtype)
    for i in range(d):
        Wz[i, i, w - 1, h - 1] = 1
    Wz -= S / f_cost.L
    Wx = D[:, :, ::-1, ::-1] / (f_cost.L * c)

    # Convolution in keras only work with kernel with shape
    # (w, h, input_filters, output_filters)
    # The convolution are reverser
    Dk = np.transpose(D, (2, 3, 0, 1))[::-1, ::-1]
    Wxk = np.transpose(Wx, (2, 3, 1, 0))[::-1, ::-1]
    Wzk = np.transpose(Wz, (2, 3, 1, 0))[::-1, ::-1]

    layer_x = Conv2D(d, Wx_size, padding="valid", data_format='channels_first',
                     use_bias=False, kernel_initializer=lambda s: Wxk,
                     trainable=False)
    layer_z = Conv2D(d, Wz_size, padding="same", data_format='channels_first',
                     use_bias=False, kernel_initializer=lambda s: Wzk,
                     trainable=False)

    # Define activation
    activation = get_soft_thresholding(lmbd / f_cost.L)

    # Define an input layer
    inputs = x = Input(shape=input_dim)
    cost_layer = get_cost_lasso_layer(x, Dk, lmbd)

    def loss_lasso(_, zk):
        return cost_layer(zk)

    # The first layer is composed by only one connection so we define it using
    # the keras API
    if z0 is None:
        z = activation(layer_x(x))
    else:
        if z0.ndim == 5:
            z0 = z0[:, :, 0]
        z = Input(shape=z0.shape[1:])
        inputs = [x, z]
        z = activation(Add()([layer_x(x), layer_z(z)]))

    # For the following layers, we define the convolution with the previous
    # layer and the input of the network and merge it for the activation layer
    cost = [cost_layer(Lambda(lambda z: 0 * z)(z)), cost_layer(z)]
    for k in range(max_iter - 1):
        z = activation(Add()([layer_z(z), layer_x(x)]))
        cost += [cost_layer(z)]

    # Construct the model
    model = Model(inputs=inputs, outputs=[z] + cost)

    if z0 is None:
        result = model.predict(X, verbose=1)
    else:
        result = model.predict([X, z0], verbose=1)

    zk, cost = result[0], result[1:]
    cost = np.mean(cost, axis=1)

    return zk, cost


def conv_fista_network(X, D, lmbd, max_iter, z0=None, verbose=0):
    """Convolutional fast ISTA for X and D implemented using keras network

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
    cost (list): cost at each iteration of the algorithm

    """
    input_dim = X.shape[1:]
    d, c = D.shape[:2]
    Wx_size = w, h = D.shape[-2:]

    assert c == X.shape[1], "mismatched dimension between input and dictionary"

    # Compute constants

    f_cost = ConvL2_z(np.zeros((1,) + input_dim), D)
    S = np.array([[[convolve2d(d1, d2, mode="full")
                   for d1, d2 in zip(dk, dl)]
                  for dk in D] for dl in D[:, :, ::-1, ::-1]]).mean(axis=2)
    Wz_size = S.shape[-2:]
    Wz = np.zeros(S.shape, dtype=D.dtype)
    for i in range(d):
        Wz[i, i, w - 1, h - 1] = 1
    Wz -= S / f_cost.L
    Wx = D[:, :, ::-1, ::-1] / (f_cost.L * c)

    # Convolution in keras only work with kernel with shape
    # (w, h, input_filters, output_filters)
    # The convolution are reverser
    Dk = np.transpose(D, (2, 3, 0, 1))[::-1, ::-1]
    Wxk = np.transpose(Wx, (2, 3, 1, 0))[::-1, ::-1]
    Wzk = np.transpose(Wz, (2, 3, 1, 0))[::-1, ::-1]

    layer_x = Conv2D(d, Wx_size, padding="valid", data_format='channels_first',
                     use_bias=False, kernel_initializer=lambda s: Wxk,
                     trainable=False)
    layer_z = Conv2D(d, Wz_size, padding="same", data_format='channels_first',
                     use_bias=False, kernel_initializer=lambda s: Wzk,
                     trainable=False)

    # Define activation
    activation = get_soft_thresholding(lmbd / f_cost.L)

    # Define an input layer
    inputs = x = Input(shape=input_dim)
    cost_layer = get_cost_lasso_layer(x, Dk, lmbd)

    def loss_lasso(_, zk):
        return cost_layer(zk)

    # The first layer is composed by only one connection so we define it using
    # the keras API

    if z0 is None:
        z = activation(layer_x(x))
    else:
        if z0.ndim == 5:
            z0 = z0[:, :, 0]
        z = Input(shape=z0.shape[1:])
        inputs = [x, z]
        z = activation(Add()([layer_x(x), layer_z(z)]))

    z_old, momentum_1 = z, 1
    momentum = 1
    cost = [cost_layer(Lambda(lambda z: 0 * z)(z)), cost_layer(z)]

    # For the following layers, we define the convolution with the previous
    # layer and the input of the network and merge it for the activation layer
    for _ in range(max_iter - 1):
        momentum = (1 + np.sqrt(1 + 4 * momentum * momentum)) / 2
        tk = (momentum_1 - 1) / momentum
        y1 = Conv2D(d, Wz_size, padding="same",
                    data_format='channels_first', use_bias=False,
                    kernel_initializer=lambda s: (1 + tk) * Wzk)(z)
        y2 = layer_x(x)
        y3 = Conv2D(d, Wz_size, padding='same',
                    data_format='channels_first', use_bias=False,
                    kernel_initializer=lambda s: -tk * Wzk)(z_old)
        y4 = layer_z(z)
        # Keep the previous values for the next layer
        z_old, momentum_1 = z, momentum

        # Update current point and store cost
        z1 = activation(Add()([y1, y2, y3]))
        z2 = activation(Add()([y4, y2]))
        c = cost_layer(z1)
        z = Lambda(lambda x: K.tf.cond(K.mean(x[2]) < K.mean(x[3]),
                                       lambda: x[0], lambda: x[1]))(
            [z1, z2, c, cost[-1]]
        )
        cost += [cost_layer(z)]

    # Construct the model
    model = Model(inputs=inputs, outputs=[z] + cost)

    if z0 is None:
        result = model.predict(X, verbose=1)
    else:
        result = model.predict([X, z0], verbose=1)

    zk, cost = result[0], result[1:]
    cost = np.mean(cost, axis=1)

    return zk, cost
