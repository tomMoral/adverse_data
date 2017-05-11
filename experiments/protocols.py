import os
import numpy as np
import matplotlib as mpl
from datetime import datetime

from .models.ista import ista, fista
from .models.ista import conv_ista_network, conv_fista_network
from .models.convolutional import conv_lista_network
from .models.utils import cost_network


def conv_network_comparison(X, D, Z, lmbd, N_layers=100, N_epochs=100,
                            ratio=1 / 3, name="cost_conv_lista"):
    """Compare the performance of ISTA, FISTA and LISTA on the convolutional
    sparse coding problem with X, D and Z.
    """
    c, p, _ = X.shape[1:]
    _layers = np.unique(np.logspace(0, np.log10(N_layers), 10).astype(int))

    N, zk_shape = Z.shape[0], Z.shape[1:]
    Dk = np.transpose(D, (2, 3, 0, 1))[::-1, ::-1]
    _cost_model = cost_network((X.shape[1:], zk_shape), Dk, lmbd)
    _cost = lambda zk: _cost_model([X, zk]).mean()

    # Extract the validation set:
    N_test = int(N * ratio)
    X_test, X_train = X[:N_test], X[N_test:]
    Z_test, Z_train = Z[:N_test], Z[N_test:]

    # Compute ISTA/FISTA
    zs, cost_fista = conv_fista_network(X=X_test, D=D, lmbd=lmbd,
                                        max_iter=N_layers, verbose=1)
    zk, cost_ista = conv_ista_network(X=X_test, D=D, lmbd=lmbd,
                                      max_iter=N_layers, verbose=1)

    # Base line
    weights = None
    c0 = _cost(0 * Z)
    results = []
    cost_curve = [c0]
    for n_layers in _layers:
        n_epochs = int(N_epochs / max(np.log2(n_layers), 1))
        print(f"Training network with {n_layers} layer for {n_epochs} epochs")
        m, loss = conv_lista_network(
            (c, p, p), D, n_layers=n_layers, activation="st", lmbd=lmbd,
            weights=weights)

        z_init = m.predict(X)
        c1 = _cost(z_init)
        assert c1 <= c0
        history = m.fit(X_train, 0 * Z_train, epochs=n_epochs,
                        validation_data=(X_test, Z_test), verbose=1)
        z_train = m.predict(X)
        c2 = _cost(z_train)
        print(f"Initial cost for {n_layers} layer: {c1}")
        print(f"Final cost for {n_layers} layer: {c2}")
        results += [(n_layers, history, c1, c2, z_train)]
        cost_curve += [c2]
        c0, weights = c2, m.get_weights()

    # Display results
    plot_result_lista(cost_ista, cost_fista, cost_curve, _layers, name)


def network_comparison(X, D, Z, lmbd, N_layers=100, N_epochs=100,
                       ratio=1 / 3, name="cost_lista"):
    """Compare the performance of ISTA, FISTA and LISTA on the sparse coding
    problem with X, D and Z.
    """
    N, p = X.shape[:2]
    _layers = np.unique(np.logspace(0, np.log10(N_layers), 10).astype(int))

    # Extract the validation set:
    N_test = int(N * ratio)
    X_test, X_train = X[:N_test], X[N_test:]
    Z_test, Z_train = Z[:N_test], Z[N_test:]

    # Compute ISTA/FISTA
    zs, cost_fista = fista(X=X_test, D=D, lmbd=lmbd, max_iter=N_layers,
                           verbose=1)
    zk, cost_ista = ista(X=X_test, D=D, lmbd=lmbd, max_iter=N_layers,
                         verbose=1)

    # Base line
    weights = None
    c0 = _cost(0 * Z)
    results = []
    cost_curve = [c0]
    for n_layers in _layers:
        n_epochs = int(N_epochs / max(np.log2(n_layers), 1))
        print(f"Training network with {n_layers} layer for {n_epochs} epochs")
        m, loss = conv_lista_network(
            (c, p, p), D, n_layers=n_layers, activation="st", lmbd=lmbd,
            weights=weights)

        z_init = m.predict(X)
        c1 = _cost(z_init)
        assert c1 <= c0
        history = m.fit(X_train, 0 * Z_train, epochs=n_epochs,
                        validation_data=(X_test, Z_test), verbose=1)
        z_train = m.predict(X)
        c2 = _cost(z_train)
        print(f"Initial cost for {n_layers} layer: {c1}")
        print(f"Final cost for {n_layers} layer: {c2}")
        results += [(n_layers, history, c1, c2, z_train)]
        cost_curve += [c2]
        c0, weights = c2, m.get_weights()

    # Display results
    plot_result_lista(cost_ista, cost_fista, cost_curve, _layers, name)


def plot_result_lista(cost_ista, cost_fista, cost_curve, _layers, name):
    """Plot the curves resulting from our computations
    """
    # Plot the result
    r = os.system('python -c "import matplotlib.pyplot as plt;plt.figure()"')
    if r != 0:
        import IPython
        IPython.embed()
        mpl.use('Agg')
    import matplotlib.pyplot as plt

    eps = 1e-6
    c_max = cost_fista[0]
    c_min = min(np.min(cost_fista), np.min(cost_ista), np.min(cost_curve))
    scale = lambda c: (c - c_min) / (c_max - c_min) + eps
    plt.semilogy(scale(cost_ista), "b", label="ISTA")
    plt.semilogy(scale(cost_fista), "g", label="FISTA")
    plt.semilogy(np.r_[0, _layers], scale(cost_curve), "r", label="LISTA")
    plt.hlines(eps, 0, _layers[-1] + 10, 'k', '--')
    plt.legend()
    exp_name = f"{name}_{datetime.now().strftime('%d%b_%Hh%M')}.pdf"
    plt.savefig(exp_name, dpi=150)
    np.save(exp_name, [cost_ista, cost_fista, cost_curve])
    plt.show()
