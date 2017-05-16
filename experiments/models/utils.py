import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Activation, ZeroPadding2D, Lambda


def soft_thresholding(x, mu):
    """Compute the soft thresholding"""
    return np.sign(x) * np.maximum(abs(x) - mu, 0)


def get_soft_thresholding_layer(mu):
    def soft_thresholding(x):
        return K.sign(x) * K.maximum(K.abs(x) - mu, 0)
    return Activation(soft_thresholding)


def sparse_coding_cost(x, D, zk, lmbd):
    err = x - zk.dot(D)
    l2 = np.sum(err * err, axis=1).mean()
    l1 = np.sum(abs(zk), axis=1).mean()
    return 1 / 2 * l2 + lmbd * l1


def get_cost_lasso_layer(X, D, lmbd):
    """Return a Layer that compute the LASSO cost for a given z

            \| X - zD \|_2^2 / 2 + lmbd \| z \|_1

    Parameters
    ----------
    X (input of the layer): data to fit with the LASSO
    D (array-like: (K, p)): dictionary used in the LASSO.
    lmbd (float): regularization parameter for the LASSO

    Return
    ------
    cost_layer (keras.Layer): compute the lasso cost for the given input zk

    Implementation note: We zero pad z to obtain the right boundary conditions,
    with the border coefficient extending the image by the kernel size.
    """
    def cost_lasso(z):
        X_rec = K.dot(z, K.constant(D))
        err = X_rec - X
        cost = K.sum(err ** 2, axis=1) / 2
        cost += lmbd * K.sum(K.abs(z), axis=1)
        return cost

    def cost_lasso_shape(input_shape):
        return ()

    cost_layer = Lambda(cost_lasso, output_shape=cost_lasso_shape)
    return cost_layer


def get_cost_conv_lasso_layer(X, D, lmbd):
    """Return a Layer that compute the convolutional LASSO cost for a given z

            \| X - z * D \|_2^2 / 2 + lmbd \| z \|_1

    Parameters
    ----------
    X (input of the layer): data to fit with the LASSO
    D (array-like: w, h, K, c): dictionary used in the LASSO. The shape is
        reversed compare to the natural shape, with the convolution axis in
        first, then the number of dictionary and finally, the channels.
    lmbd (float): regularization parameter for the LASSO

    Return
    ------
    cost_layer (keras.Layer): compute the lasso cost for the given input zk

    Implementation note: We zero pad z to obtain the right boundary conditions,
    with the border coefficient extending the image by the kernel size.
    """
    def cost_lasso(zk):
        padding = (D.shape[0] // 2, D.shape[1] // 2)
        zp = ZeroPadding2D(padding=padding, data_format='channels_first')(zk)

        X_rec = K.conv2d(zp, K.constant(D), padding='same',
                         data_format='channels_first')
        err = X_rec - X
        cost = K.sum(K.mean(err ** 2, axis=(0, 1))) / 2
        cost += lmbd * K.sum(K.mean(K.abs(zk), axis=0))
        return cost

    def cost_lasso_shape(input_shape):
        return ()

    cost_layer = Lambda(cost_lasso, output_shape=cost_lasso_shape)
    return cost_layer


def cost_network(input_dims, D, lmbd, conv):
    x = Input(shape=input_dims[0])
    z = Input(shape=input_dims[1])
    if conv:
        cost = get_cost_conv_lasso_layer(x, D, lmbd)(z)
    else:
        cost = get_cost_lasso_layer(x, D, lmbd)(z)

    cost_network = Model(inputs=[x, z], outputs=cost)
    _cost = cost_network.predict
    return _cost


def cost_conv_network(input_dims, D, lmbd):
    x = Input(shape=input_dims[0])
    z = Input(shape=input_dims[1])
    cost = get_cost_conv_lasso_layer(x, D, lmbd)(z)

    cost_network = Model(inputs=[x, z], outputs=cost)
    _cost = cost_network.predict
    return _cost
