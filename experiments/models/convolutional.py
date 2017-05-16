import keras
import numpy as np
from scipy.signal import convolve2d

from keras.backend import get_session
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers.merge import Add
from ._cost import ConvL2_z
from .utils import get_soft_thresholding_layer, get_cost_conv_lasso_layer


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


def convolutional_lista_network(input_dim, d, kernel_size, num_classes,
                                n_layers=10, activation="relu", alpha=1e-5,
                                D=None, lmbd=1):
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
        activation = get_soft_thresholding_layer(lmbd)
    else:
        activation = Activation(activation)

    x = Input(shape=input_dim)
    z = Conv2D(d, kernel_size, padding="same", data_format='channels_first')(x)
    z = activation(z)
    for _ in range(n_layers - 1):
        y1 = Conv2D(d, kernel_size, padding="same",
                    data_format='channels_first')(z)
        y2 = Conv2D(d, kernel_size, padding="same",
                    data_format='channels_first')(x)
        z = activation(keras.layers.add([y1, y2]))
    h = Flatten()(z)
    y_pred = Dense(num_classes, activation='softmax')(h)
    model = Model(inputs=x, outputs=y_pred)

    if D is not None:
        # Convolution in keras only work with kernel with shape (w, c, ic, oc)

        Dk = np.transpose(D, (2, 3, 0, 1))
        rec = K.conv2d(z, K.constant(Dk), padding='same',
                       data_format='channels_first')
        diff = model.input - rec
        cost = K.sum(diff * diff)
        cost += lmbd * K.sum(K.abs(z))

        def metric_cost(y, yp):
            return cost

        def loss(y, yp):
            return keras.metrics.categorical_crossentropy(y, yp) + alpha * cost
    else:
        def metric_cost(y, yp):
            return keras.metrics.categorical_crossentropy(y, yp)

        def loss(y, yp):
            return keras.metrics.categorical_crossentropy(y, yp)

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss=loss,
                  optimizer=opt, metrics=[keras.metrics.categorical_accuracy,
                                          metric_cost])

    return model


def alexnet(input_shape, num_classes):
    """CIFAR10 model resolution
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


def lista_conv_network(input_dim, D, n_layers=10, activation="st",
                       lmbd=1, weights=None):
    """Construct a convolutional LISTA network with n_layers.

    Parameters
    ----------
    input_dim (tuple): size of the input for this network
    D (array-like): dictionary used in LISTA
    lmbd (float: 1): regularization parameter for the sparse coding
    n_layers (int:10): number of layer in the network
    activation (str:None)

    Return args
    -----------
    model: a keras.model containing the network.
    """

    D = np.array(D)
    d, c = D.shape[:2]
    Wx_size = w, h = D.shape[-2:]

    assert input_dim[0] == c, "mismatched dimension between input and dictionary"

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

    if weights is None:
        weights = [(Wxk, Wzk)] * n_layers
    elif len(weights) < n_layers:
        weights = list(weights) + [(Wxk, Wzk)] * (n_layers - len(weights))

    if activation == "st":
        activation = get_soft_thresholding_layer(lmbd / f_cost.L)
    else:
        activation = Activation(activation)

    # Define an input layer
    x = Input(shape=input_dim)

    # Compute the loss for our network using the LASSO minimization problem
    cost_layer = get_cost_conv_lasso_layer(x, Dk, lmbd)

    # The first layer is composed by only one connection so we define it using
    # the keras API
    wx0, _ = weights[0]
    z = Conv2D(d, Wx_size, padding="valid", data_format='channels_first',
               use_bias=False, kernel_initializer=lambda s: wx0)(x)
    z = activation(z)

    # For the following layers, we define the convolution with the previous
    # layer and the input of the network and merge it for the activation layer
    for k in range(n_layers - 1):
        wx0, wz0 = weights[k + 1]
        y1 = Conv2D(d, Wz_size, padding="same",
                    data_format='channels_first', use_bias=False,
                    kernel_initializer=lambda s: wz0)(z)
        y2 = Conv2D(d, Wx_size, padding="valid",
                    data_format='channels_first', use_bias=False,
                    kernel_initializer=lambda s: wx0)(x)
        z = activation(Add()([y1, y2]))

    # Construct the model

    model = Model(inputs=x, outputs=[z, cost_layer(z)])

    def loss(y, yp):
        print("Shape yp[0", yp.get_shape())
        return cost_layer(z)

    opt = keras.optimizers.rmsprop(lr=0.0001 / n_layers, decay=1e-6)
    model.compile(loss=loss, optimizer=opt)

    model.Wz = Wz
    model.Wx = Wx

    def export_weights():
        w = [(model.weights[0].eval(get_session()), None)]
        for wz, wx in zip(model.weights[1::2], model.weights[2::2]):
            w += [(wx.eval(get_session()), wz.eval(get_session()))]
        return w

    model.export_weights = export_weights

    return model
