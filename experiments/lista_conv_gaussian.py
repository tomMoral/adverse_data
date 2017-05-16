from .datasets.dictionaries import create_gaussian_conv_dictionary
from .datasets.sparse_conding import conv_sparse_code

from .protocols import conv_network_comparison

# Model parameters
N = 9000
d = 10
w = 64
k = 5
c = 3

rho = .02
lmbd = .01
sigma = 10

N_layers = 100
N_epochs = 500

# internal
data_shape = (N, w, w, c)
num_classes = 2

D = create_gaussian_conv_dictionary(d, (c, k, k))
z, data = conv_sparse_code(N, w, D, rho=rho, sigma=sigma)
conv_network_comparison(data, D, z, lmbd, N_layers=N_layers, N_epochs=N_epochs)


raise SystemExit(0)
