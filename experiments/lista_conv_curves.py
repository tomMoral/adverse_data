import numpy as np
import matplotlib.pyplot as plt

from .models.ista import ista_conv, fista_conv
from .models.ista_network import ista_conv_network, fista_conv_network
from .models.convolutional import convolutional_lista_network
from .datasets.dictionaries import create_gaussian_conv_dictionary
from .datasets.sparse_coding import conv_sparse_code

# Model parameters
N = 100
d = 10
w = 64
k = 5
c = 3

N_layers = 4

rho = .02
lmbd = 1
sigma = 10

num_classes = 2

# internal
D = create_gaussian_conv_dictionary(d, (c, k, k))
z, data = conv_sparse_code(N, w, D, rho=rho, sigma=sigma)

model = convolutional_lista_network((c, w, w), d, (k, k), num_classes,
                                    n_layers=10, alpha=1e-8,
                                    activation='relu', D=D, lmbd=lmbd)

# Base line
max_iter = 100
zs, cost_fista = fista_conv_network(X=data, D=D, lmbd=lmbd, max_iter=max_iter,
                                    verbose=1)
print(f"Progress: {1.:7.2%} - {cost_fista[-1]:15.6f}")
zk, cost_ista = ista_conv_network(X=data, D=D, lmbd=lmbd, max_iter=max_iter,
                                  verbose=1)
_, cost_ista2 = ista_conv(data, D, lmbd, max_iter, verbose=1)
_, cost_fista2 = fista_conv(data, D, lmbd, max_iter, verbose=1)

eps = 1e-6
c_max = cost_fista[0]
c_min = min(np.min(cost_fista), np.min(cost_ista))
scale = lambda c: (c - c_min) / (c_max - c_min) + eps
plt.loglog(scale(cost_ista), "b", label="ISTA_network")
plt.loglog(scale(cost_fista), "r", label="FISTA_network")
plt.loglog(scale(cost_ista2), "c", label="ISTA")
plt.loglog(scale(cost_fista2), "m", label="FISTA")
plt.hlines(eps, 1, max_iter + 10, 'k', '--')
plt.legend()
plt.show()


raise SystemExit(0)
