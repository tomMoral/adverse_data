import keras
import numpy as np
from keras import backend as K
from scipy.signal import convolve2d

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D
from keras.layers import merge
from keras.layers.merge import Add
from ._cost import ConvL2_z


def get_soft_thresholding(mu):
    def soft_thresholding(x):
        return K.sign(x) * K.maximum(K.abs(x) - mu, 0)
    return Activation(soft_thresholding)


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
        activation = get_soft_thresholding(lmbd)
    else:
        activation = Activation(activation)

    x = Input(shape=input_dim)
    z = Conv2D(d, kernel_size, padding="same", activation=activation,
               data_format='channels_first')(x)
    for _ in range(n_layers - 1):
        y1 = Conv2D(d, kernel_size, padding="same",
                    data_format='channels_first')(z)
        y2 = Conv2D(d, kernel_size, padding="same",
                    data_format='channels_first')(x)
        z = activation(keras.layers.add([y1, y2]))
    h = Flatten()(z)
    y_pred = Dense(num_classes, activation='softmax')(h)
    print(y_pred.get_shape)
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


def conv_lista_network(input_dim, D, n_layers=10, activation="st",
                       lmbd=1):
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
    p, (d, c) = input_dim[0], D.shape[:2]
    Wx_size = w, h = D.shape[-2:]

    assert p == D.shape[1], "mismatched dimension between input and dictionary"

    # Compute constants

    f_cost = ConvL2_z(np.zeros((10,) + input_dim), D)
    S = np.array([[[convolve2d(d1, d2, mode="full")
                   for d1, d2 in zip(dk, dl)]
                  for dk in D] for dl in D[:, :, ::-1, ::-1]]).mean(axis=2)
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

    if activation == "st":
        activation = get_soft_thresholding(lmbd / f_cost.L)
    else:
        activation = Activation(activation)

    def Wz_initializer(shape, dtype=None):
        return Wzk

    def Wx_initializer(shape, dtype=None):
        return Wxk
    Wz_size = Wz.shape[-2:]

    # Define an input layer
    x = Input(shape=input_dim)

    # The first layer is composed by only one connection so we define it using
    # the keras API
    z = Conv2D(d, Wx_size, padding="valid", activation=activation,
               data_format='channels_first', use_bias=False,
               kernel_initializer=Wx_initializer)(x)

    # For the following layers, we define the convolution with the previous
    # layer and the input of the network and merge it for the activation layer
    for _ in range(n_layers - 1):
        y1 = Conv2D(d, Wz_size, padding="same",
                    data_format='channels_first', use_bias=False,
                    kernel_initializer=Wz_initializer)(z)
        y2 = Conv2D(d, Wx_size, padding="valid",
                    data_format='channels_first', use_bias=False,
                    kernel_initializer=Wx_initializer)(x)
        z = activation(Add()([y1, y2]))

    # Compute the loss for our network using the LASSO minimization problem
    # We zero pad z to obtain the right boundary conditions, with the
    # border coefficient extending the image by the kernel size.
    def loss_lasso(_, zk):
        padding = (Dk.shape[0] // 2, Dk.shape[1] // 2)
        zp = ZeroPadding2D(padding=padding, data_format='channels_first')(zk)

        x_rec = K.conv2d(zp, K.constant(Dk), padding='same',
                         data_format='channels_first')
        err = x_rec - x
        cost = K.sum(K.mean(err ** 2, axis=1)) / 2
        cost += lmbd * K.sum(K.abs(zk))
        return cost

    # padding = (Dk.shape[0] // 2, Dk.shape[1] // 2)
    # zp = ZeroPadding2D(padding=padding, data_format='channels_first')(z)
    # x_rec = Conv2D(p, Wx_size, padding='same',
    #                data_format='channels_first', use_bias=False,
    #                kernel_initializer=lambda a: Dk)(zp)

    model = Model(inputs=x, outputs=z)
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss=loss_lasso, optimizer=opt)

    model.Wz = Wz
    model.Wx = Wx

    return model, loss_lasso, []
