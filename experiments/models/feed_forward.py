import keras
import numpy as np
from keras import backend as K

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input


def get_soft_thresholding(lmbd):
    def soft_thresholding(x):
        return K.relu(x - lmbd) - K.relu(-x - lmbd)


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


def lista_network(input_dim, d, num_classes, n_layers=10,
                  activation="relu", lmbd=.1, D=None):
    """Construct LISTA like network with n_layers.

    Parameters
    ----------
    input_dim (int): size of the input for this network
    d (int): number of dictionary used in LISTA
    num_classes (int): number of output
    n_layers (int:10): number of layer to add in the network
    activation (str:None)

    Return args
    -----------
    model: a keras.model containing the network.
    """
    if activation == "st":
        activation = get_soft_thresholding(lmbd)
    else:
        activation = Activation(activation)

    x = Input(shape=input_dim)
    z = Dense(d, activation=activation)(x)
    for _ in range(n_layers - 1):
        h = keras.layers.add([Dense(d)(x), Dense(d)(z)])
        z = activation(h)
    y_pred = Dense(num_classes, activation='softmax')(z)
    model = Model(inputs=x, outputs=y_pred)

    if D is not None:
        cost = keras.losses.mean_squared_error(
            model.input, K.dot(z, K.constant(D)))
        cost += lmbd * K.sum(K.abs(z))

        def metric_cost(y, yp):
            return cost

        def loss(y, yp):
            return keras.metrics.categorical_crossentropy(y, yp) + cost / 10000

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss=loss,
                  optimizer=opt, metrics=[keras.metrics.categorical_accuracy,
                                          metric_cost])

    return model
