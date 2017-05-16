import keras
import numpy as np

from keras.backend import get_session
from keras.models import Sequential, Model
from keras.layers import Dense, Add, Input

from ._cost import LossL2
from .utils import get_cost_lasso_layer, get_soft_thresholding_layer


def dummy_feedforward_network(input_dim):
    """Construct a dummy 2 layer feedforward neural network for testing.

    Parameters
    ----------
    input_dim (int): size of the input for this network

    Return args
    -----------
    model: a keras.model containing the network.
    """
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=input_dim))
    model.add(Dense(10, activation='softmax'))

    return model


def lista_network(input_dim, D, n_layers=10, activation="st", lmbd=.1,
                  weights=None):
    """Construct LISTA like network with n_layers.

    Parameters
    ----------
    input_dim (int): size of the input for this network
    D (array-like - (K, p)): dictionary to compute the sparse code.
        There is K atoms in a p-dimensional space.
    n_layers (int:10): number of layer to add in the network
    activation (str:None): activation function, for LISTA, this should be "st"

    Return args
    -----------
    model: a keras.model containing the network.
    """

    K, p = D.shape
    assert input_dim == p

    f_cost = LossL2(np.zeros(shape=(1, p)), D)
    Wz = np.eye(K, dtype=D.dtype) - D.dot(D.T) / f_cost.L
    Wx = D.T / f_cost.L
    if weights is None:
        weights = [(Wx, Wz)] * n_layers
    elif len(weights) < n_layers:
        weights = list(weights) + [(Wx, Wz)] * (n_layers - len(weights))

    # Define some layers to build the neural network
    x = Input(shape=(input_dim, ))
    cost_layer = get_cost_lasso_layer(x, D, lmbd)
    activation = get_soft_thresholding_layer(lmbd / f_cost.L)

    wx0, _ = weights[0]
    z = Dense(K, use_bias=False, kernel_initializer=lambda s: wx0)(x)
    z = activation(z)
    for k in range(n_layers - 1):
        wxk, wzk = weights[k + 1]
        y1 = Dense(K, use_bias=False, kernel_initializer=lambda s: wzk)(z)
        y2 = Dense(K, use_bias=False, kernel_initializer=lambda s: wxk)(x)
        z = activation(Add()([y1, y2]))

    model = Model(inputs=x, outputs=z)

    def loss(y, yp):
        return cost_layer(z)

    opt = keras.optimizers.rmsprop(lr=0.0001 / n_layers, decay=1e-6)
    model.compile(loss=loss, optimizer=opt)

    def export_weights():
        w = [(model.weights[0].eval(get_session()), None)]
        for wz, wx in zip(model.weights[1::2], model.weights[2::2]):
            w += [(wx.eval(get_session()), wz.eval(get_session()))]
        return w

    model.export_weights = export_weights

    return model
