import keras
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft2 as fft, irfft2 as ifft

from .models.ista import ConvL2_z
from .models.ista import conv_ista_network, conv_fista_network
from .models.convolutional import conv_lista_network
from .models.convolutional import convolutional_lista_network
from .datasets.dictionaries import create_gaussian_conv_dictionary
from .datasets.labels import sign_test_labels, parity_test_labels
from .datasets.sparse_conding import conv_sparse_code

from .protocols import conv_network_comparison

# Model parameters
N = 3000
d = 10
w = 64
k = 5
c = 3

rho = .02
lmbd = 1
sigma = 10

N_layers = 100
N_epochs = 500

# internal
data_shape = (N, w, w, c)
num_classes = 2

D = create_gaussian_conv_dictionary(d, (c, k, k))
z, data = conv_sparse_code(N, w, D, rho=rho, sigma=sigma)

model = convolutional_lista_network((c, w, w), d, (k, k), num_classes,
                                    n_layers=10, alpha=1e-8,
                                    activation='relu', D=D, lmbd=lmbd)

conv_network_comparison(data, D, z, lmbd, N_layers=N_layers, N_epochs=N_epochs)


raise SystemExit(0)


y = parity_test_labels(zs)
y_pred = parity_test_labels(zk)
labels = keras.utils.to_categorical(y, num_classes=num_classes)

print("Random performance: {:7.2%}".format(y.mean()))
print("Baseline performance: {:7.2%}".format((y_pred == y).mean()))
