import numpy as np
import matplotlib.pyplot as plt

from .models.ista import ista, fista
from .models.ista_network import ista_network, fista_network
from .models.feed_forward import lista_network
from .datasets.dictionaries import create_gaussian_dictionary
from .datasets.sparse_coding import sparse_code

# Model parameters
N = 1000
K = 100
p = 64

N_layers = 4

rho = .02
lmbd = 1
sigma = 10

# Generate the data
D = create_gaussian_dictionary(K, p)
z, data = sparse_code(N, D, rho=rho, sigma=sigma)

model = lista_network(p, D, n_layers=10, activation='st', lmbd=lmbd)

# Base line
max_iter = 100
zs, cost_fista = fista_network(X=data, D=D, lmbd=lmbd, max_iter=max_iter,
                               verbose=1)
zk, cost_ista = ista_network(X=data, D=D, lmbd=lmbd, max_iter=max_iter,
                             verbose=1)
_, cost_ista2 = ista(data, D, lmbd, max_iter, verbose=1)
_, cost_fista2 = fista(data, D, lmbd, max_iter, verbose=1)

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
