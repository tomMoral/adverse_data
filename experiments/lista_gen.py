import keras
import numpy as np

from .models.feed_forward import lista_network
from .datasets import threadsafe_generator
from .datasets.dictionaries import create_adversarial_dictionary

# Model parameters
N = 20000
N_test = 10000
K = 130
p = 64
batch_size = 32

rho = .1

# internal
data_shape = (N, p)
num_classes = 2


model = lista_network(p, K, num_classes, n_layers=10, lmbd=.0001,
                      activation='relu')

# Define the dataset
D = np.random.normal(size=(K, p))
# D = create_adversarial_dictionary(K, p)

@threadsafe_generator
def generator():
    while True:
        z = (np.random.rand(batch_size, K) < rho).astype(np.float)
        z *= np.random.normal(size=(batch_size, K))

        x = z.dot(D)
        y = [zz[zz > 0].sum() > - zz[zz < 0].sum() for zz in z]
        labels = keras.utils.to_categorical(y)
        yield x, labels
