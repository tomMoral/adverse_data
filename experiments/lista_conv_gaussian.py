import keras
import IPython
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from numpy.fft import rfft2 as fft, irfft2 as ifft

from .models.ista import ConvL2_z
from .models.ista import ista_conv, fista_conv
from .models.convolutional import conv_lista_network
from .models.convolutional import convolutional_lista_network
from .datasets.dictionaries import create_gaussian_conv_dictionary
from .datasets.labels import sign_test_labels, parity_test_labels

# Model parameters
N = 100
d = 10
p = 64
k = 5
N_layers = 100
c = 3

rho = .02
lmbd = 1

# internal
data_shape = (N, p, p, c)
num_classes = 2

fft_shape = p + k - 1, p + k - 1

D = create_gaussian_conv_dictionary(d, (c, k, k))
D_fft = fft(D, s=fft_shape)

model = convolutional_lista_network((c, p, p), d, (k, k), num_classes,
                                    n_layers=10, alpha=1e-8,
                                    activation='relu', D=D, lmbd=lmbd)

# Define the dataset
w = h = p
# generate z support
print("Generate z support")
z = (np.random.rand(N, d, w - k + 1, h - k + 1) < rho).astype(np.float)
print("Generate z coefficients")
z *= 10 * np.random.normal(size=(N, d, w - k + 1, h - k + 1))

print("Non-Zero  in z:", 1 - np.isclose(z, 0).sum() / z.size)

z_fft = fft(z, s=fft_shape)
data = ifft((D_fft[None] * z_fft[:, :, None]).sum(axis=1))[:, :, :p, :p]

# Base line
max_iter = N_layers
m, loss, md = conv_lista_network((c, p, p), D, n_layers=max_iter,
                                 activation="st", lmbd=lmbd)

zs, cost_fista = fista_conv(X=data, D=D, lmbd=lmbd, max_iter=max_iter)
zk, cost_ista = ista_conv(X=data, D=D, lmbd=lmbd, max_iter=max_iter)
zs, zk = zs[:, :, 0], zk[:, :, 0]
zk2 = m.predict(data, batch_size=len(data))

c_model = m.evaluate(data, 0 * zs, batch_size=100)


f_cost = ConvL2_z(data, D)


def _cost(z):
    return f_cost(z) + lmbd * abs(z).sum()


print(f"Linf: {abs(zk-zs).max()}")
print(f"Linf: {abs(zk-zk2).sum()}")
print(f"Cost: ista {_cost(zk)}, lista {_cost(zk2)}, model: {c_model}")
assert np.allclose(zk, zk2, atol=1e-5)
c_model = _cost(zk2)

assert np.isclose(c_model, _cost(zk2))

eps = 1e-5
c_min = min(np.min(cost_fista), np.min(cost_ista)) - eps
plt.semilogy(cost_ista - c_min)
plt.semilogy(cost_fista - c_min)
plt.hlines(eps, 0, max_iter + 10, 'k', '--')
plt.hlines(c_model - c_min, 0, max_iter + 10, 'r', '--')
plt.show()
raise SystemExit(0)


y = parity_test_labels(zs)
y_pred = parity_test_labels(zk)
labels = keras.utils.to_categorical(y, num_classes=num_classes)

print("Random performance: {:7.2%}".format(y.mean()))
print("Baseline performance: {:7.2%}".format((y_pred == y).mean()))

