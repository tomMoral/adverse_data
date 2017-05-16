from .datasets.dictionaries import create_adversarial_dictionary
from .datasets.sparse_coding import sparse_code
from .protocols import network_comparison

# Model parameters
N = 90000
K = 256
p = 50

rho = .05
lmbd = .01
sigma = 10

N_layers = 200
N_epochs = 800

D = create_adversarial_dictionary(K, p)
z, data = sparse_code(N, D, rho=rho, sigma=sigma)

network_comparison(data, D, z, lmbd, N_layers=N_layers, N_epochs=N_epochs,
                   name="lista_adverse")


raise SystemExit(0)
