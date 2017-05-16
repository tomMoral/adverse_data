import numpy as np
from scipy.signal import convolve2d

# Import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, Add, Lambda, Dense

from ._cost import ConvL2_z, LossL2
from .utils import get_soft_thresholding_layer
from .utils import get_cost_lasso_layer, get_cost_conv_lasso_layer


def ista_network(X, D, lmbd, max_iter, z0=None, verbose=0):
    """ISTA for X and D implemented using keras network

    Parameters
    ----------
    X (array-like - (N, p)): signal to compute the sparse code.
        There is N samples with p dimensional 2d signals of size wxh.
    D (array-like - (K, p)): dictionary to compute the sparse code.
        There is K atoms in a p-dimensional space.
    lmbd (float): regularization parameter for the sparse coding
    max_iter (int): number of iteration for FISTA

    Return
    ------
    zk (array-like - (N, K, w-wk+1, h-hk+1)): code for the given signals
    cost (list): cost at each iteration of the algorithm

    """
    d, p = D.shape

    assert X.shape[1] == p, "mismatched dimension between input and dictionary"

    # Compute constants
    f_cost = LossL2(np.zeros(shape=(1, p)), D)
    Wz = np.eye(d, dtype=D.dtype) - D.dot(D.T) / f_cost.L
    Wx = D.T / f_cost.L

    layer_x = Dense(d, use_bias=False, kernel_initializer=lambda s: Wx,
                    trainable=False)
    layer_z = Dense(d, use_bias=False, kernel_initializer=lambda s: Wz,
                    trainable=False)

    # Define an input layer
    inputs = x = Input(shape=(p,))
    cost_layer = get_cost_lasso_layer(x, D, lmbd)
    activation = get_soft_thresholding_layer(lmbd / f_cost.L)

    # The first layer is composed by only one connection so we define it using
    # the keras API
    if z0 is None:
        z = activation(layer_x(x))
    else:
        z = Input(shape=(d,))
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


def fista_network(X, D, lmbd, max_iter, z0=None, verbose=0):
    """Fast ISTA for X and D implemented using keras network

    Parameters
    ----------
    X (array-like - (N, p)): signal to compute the sparse code.
        There is N samples with p dimensions.
    D (array-like - (K, p)): dictionary to compute the sparse code.
        There is K atoms in a p-dimensional space.
    lmbd (float): regularization parameter for the sparse coding
    max_iter (int): number of iteration for FISTA

    Return
    ------
    zk (array-like - (N, K)): code for the given signals
    cost (list): cost at each iteration of the algorithm

    """
    d, p = D.shape

    assert X.shape[1] == p, "mismatched dimension between input and dictionary"

    # Compute constants
    f_cost = LossL2(np.zeros(shape=(1, p)), D)
    Wz = np.eye(d, dtype=D.dtype) - D.dot(D.T) / f_cost.L
    Wx = D.T / f_cost.L

    layer_x = Dense(d, use_bias=False, kernel_initializer=lambda s: Wx,
                    trainable=False)
    layer_z = Dense(d, use_bias=False, kernel_initializer=lambda s: Wz,
                    trainable=False)

    # Define activation
    activation = get_soft_thresholding_layer(lmbd / f_cost.L)

    # Define an input layer
    inputs = x = Input(shape=(p,))
    cost_layer = get_cost_lasso_layer(x, D, lmbd)

    # The first layer is composed by only one connection but we add a z input
    # if z0 is provided to allow computing from a given starting point
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
        y1 = Dense(d, use_bias=False, trainable=False,
                   kernel_initializer=lambda s: (1 + tk) * Wz)(z)
        y2 = layer_x(x)
        y3 = Dense(d, use_bias=False, trainable=False,
                   kernel_initializer=lambda s: -tk * Wz)(z_old)
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


def ista_conv_network(X, D, lmbd, max_iter, z0=None, verbose=0):
    """Convolutional ISTA for X and D implemented using keras network

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
    activation = get_soft_thresholding_layer(lmbd / f_cost.L)

    # Define an input layer
    inputs = x = Input(shape=input_dim)
    cost_layer = get_cost_conv_lasso_layer(x, Dk, lmbd)

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


def fista_conv_network(X, D, lmbd, max_iter, z0=None, verbose=0):
    """Convolutional fast ISTA for X and D implemented using keras network

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
    activation = get_soft_thresholding_layer(lmbd / f_cost.L)

    # Define an input layer
    inputs = x = Input(shape=input_dim)
    cost_layer = get_cost_conv_lasso_layer(x, Dk, lmbd)

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
