import keras
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft2 as fft, irfft2 as ifft

from .models.ista import ConvL2_z
from .models.ista import ista_conv, fista_conv, conv_ista_network
from .models.convolutional import conv_lista_network
from .models.convolutional import convolutional_lista_network
from .datasets.dictionaries import create_gaussian_conv_dictionary
from .datasets.labels import sign_test_labels, parity_test_labels

# Model parameters
N = 3000
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


def _cost(z):
    f_cost = ConvL2_z(data, D)
    return f_cost(z) + lmbd * abs(z).mean(axis=0).sum()


# Base line
weights = None
N_epochs = 800
c0 = _cost(0 * z)
results = []
curve_cost = [c0]
_layers = np.unique(np.logspace(0, np.log10(N_layers), 10).astype(int))
for n_layers in _layers:
    n_epochs = N_epochs // max(int(np.log(n_layers)), 1)
    print(f"Training network with {n_layers} layer for {n_epochs} epochs")
    m, loss = conv_lista_network((c, p, p), D, n_layers=n_layers,
                                 activation="st", lmbd=lmbd, weights=weights)

    z_init = m.predict(data, batch_size=128)
    c1 = _cost(z_init)
    assert c1 <= c0
    history = m.fit(data, 0 * z, epochs=n_epochs,
                    validation_split=1 / 3, verbose=1)
    z_train = m.predict(data, batch_size=128)
    c2 = _cost(z_train)
    print(f"Initial cost for {n_layers} layer: {c1}")
    print(f"Final cost for {n_layers} layer: {c2}")
    results += [(n_layers, history, c1, c2, z_train)]
    curve_cost += [c2]
    c0, weights = c2, m.get_weights()

max_iter = 100
zs, cost_fista = fista_conv(X=data, D=D, lmbd=lmbd, max_iter=max_iter,
                            verbose=1)
zk, cost_ista = ista_conv(X=data, D=D, lmbd=lmbd, max_iter=N_layers,
                          verbose=1)
zs, zk = zs[:, :, 0], zk[:, :, 0]


import IPython
IPython.embed()
eps = 1e-3
c_min = min(np.min(cost_fista), np.min(cost_ista), np.min(curve_cost)) - eps
plt.semilogy(cost_ista - c_min, 'b', label="ISTA")
plt.semilogy(cost_fista - c_min, 'r', label="FISTA")
plt.semilogy(np.r_[0, _layers], curve_cost - c_min, 'g', label="LISTA")
plt.hlines(eps, 0, max_iter + 10, 'k', '--')
plt.legend()
plt.savefig("cost_lista_10mai.pdf", dpi=150)
np.save("cost_lista_10mai", [cost_ista, cost_fista, curve_cost])
plt.show()


raise SystemExit(0)


y = parity_test_labels(zs)
y_pred = parity_test_labels(zk)
labels = keras.utils.to_categorical(y, num_classes=num_classes)

print("Random performance: {:7.2%}".format(y.mean()))
print("Baseline performance: {:7.2%}".format((y_pred == y).mean()))

