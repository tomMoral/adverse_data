import numpy as np
import keras

from .models.ista import ista, fista
from .models.feed_forward import lista_network
from .datasets.dictionaries import create_adversarial_dictionary
from .datasets.labels import sign_test_labels, parity_test_labels

# Model parameters
N = 100000
K = 130
p = 64
N_layers = 4

rho = .2

# internal
data_shape = (N, p)
num_classes = 2

D = create_adversarial_dictionary(K, p)


model = lista_network((p,), K, num_classes, n_layers=N_layers, lmbd=1,
                      activation='relu', D=D)

# Define the dataset
z = (np.random.rand(N, K) < rho).astype(np.float)
z *= 1 * np.random.normal(size=(N, K))
data = z.dot(D)

# Base line
zk, cost = ista(data, N_layers, D, 1)
zs, cost_full = fista(data, 100, D, 1)

y = parity_test_labels(zs)
y_pred = parity_test_labels(zk)
labels = keras.utils.to_categorical(y, num_classes=num_classes)
