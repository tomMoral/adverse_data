import os
import numpy as np
import matplotlib as mpl
from datetime import datetime
from keras.callbacks import EarlyStopping

from .models.ista_network import ista_network, fista_network
from .models.ista_network import ista_conv_network, fista_conv_network
from .models.feed_forward import lista_network
from .models.convolutional import lista_conv_network
from .models.utils import cost_network, cost_conv_network
from .datasets.labels import parity_test_labels



def conv_network_comparison(X, D, Z, lmbd, N_layers=100, N_epochs=100,
                            ratio=1 / 3, name="cost_conv_lista"):
    """Compare the performance of ISTA, FISTA and LISTA on the convolutional
    sparse coding problem with X, D and Z.
    """
    c, p, _ = X.shape[1:]
    _layers = np.unique(np.logspace(0, np.log10(N_layers), 10).astype(int))

    N, zk_shape = Z.shape[0], Z.shape[1:]
    Dk = np.transpose(D, (2, 3, 0, 1))[::-1, ::-1]
    _cost_model = cost_conv_network((X.shape[1:], zk_shape), Dk, lmbd)
    _cost = lambda x, z: _cost_model([x, z]).mean()

    # Extract the validation set:
    N_test = int(N * ratio)
    X_test, X_train = X[:N_test], X[N_test:]
    Z_test, Z_train = Z[:N_test], Z[N_test:]

    # Compute ISTA/FISTA
    zs, cost_fista = fista_conv_network(X=X_test, D=D, lmbd=lmbd,
                                        max_iter=N_layers, verbose=1)
    zk, cost_ista = ista_conv_network(X=X_test, D=D, lmbd=lmbd,
                                      max_iter=N_layers, verbose=1)

    # Base line
    weights = None
    c0 = _cost(X_test, 0 * Z_test)
    results = []
    cost_curve = [c0]
    for n_layers in _layers:
        n_epochs = int(N_epochs / max(np.log2(n_layers), 1))
        print(f"Training network with {n_layers} layer for {n_epochs} epochs")
        m = lista_conv_network(
            (c, p, p), D, n_layers=n_layers, activation="st", lmbd=lmbd,
            weights=weights)

        zk = m.predict(X_test)
        c1 = _cost(X_test, zk)
        assert c1 <= c0
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-7,
                                       patience=10, verbose=1)
        history = m.fit(X_train, 0 * Z_train, epochs=n_epochs, verbose=1,
                        validation_data=(X_test, Z_test),
                        callbacks=[early_stopping])
        zk = m.predict(X_test)
        c2 = _cost(X_test, zk)
        print(f"Initial cost for {n_layers} layer: {c1}")
        print(f"Final cost for {n_layers} layer: {c2}")
        results += [(n_layers, history, c1, c2, zk)]
        cost_curve += [c2]
        c0, weights = c2, m.export_weights()

    # Display results
    plot_result_lista(cost_ista, cost_fista, cost_curve, _layers, name)


def network_comparison(X, D, Z, lmbd, N_layers=100, N_epochs=100,
                       ratio=1 / 3, name="cost_lista"):
    """Compare the performance of ISTA, FISTA and LISTA on the sparse coding
    problem with X, D and Z.
    """
    N, p = X.shape[:2]
    _layers = np.unique(np.logspace(0, np.log10(N_layers), 10).astype(int))

    # Define the _cost
    _cost_model = cost_network((X.shape[1:], Z.shape[1:]), D, lmbd, False)
    _cost = lambda x, z: _cost_model([x, z]).mean()

    # Extract the validation set:
    N_test = int(N * ratio)
    X_test, X_train = X[:N_test], X[N_test:]
    Z_test, Z_train = Z[:N_test], Z[N_test:]

    # Compute ISTA/FISTA
    zs, cost_fista = fista_network(X=X_test, D=D, lmbd=lmbd, max_iter=N_layers,
                                   verbose=1)
    zk, cost_ista = ista_network(X=X_test, D=D, lmbd=lmbd, max_iter=N_layers,
                                 verbose=1)

    # Base line
    weights = None
    c0 = _cost(X_test, 0 * Z_test)
    results = []
    cost_curve = [c0]
    cost_curve_mix = [c0]
    for n_layers in _layers:
        n_epochs = int(N_epochs / max(np.log2(n_layers), 1))
        print(f"Training network with {n_layers} layer for {n_epochs} epochs")
        m = lista_network(p, D, n_layers=n_layers, activation="st",
                          lmbd=lmbd, weights=weights)

        zk = m.predict(X_test)
        c1 = _cost(X_test, zk)
        try:
            assert c1.mean() <= c0
        except AssertionError:
            import IPython
            IPython.embed()
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-7,
                                       patience=20, verbose=1)
        history = m.fit(X_train, 0 * Z_train, epochs=n_epochs, verbose=1,
                        validation_data=(X_test, Z_test),
                        callbacks=[early_stopping])
        zk = m.predict(X_test)
        c2 = _cost(X_test, zk)
        print(f"Initial cost for {n_layers} layer: {c1}")
        print(f"Final cost for {n_layers} layer: {c2}")
        results += [(n_layers, history, c1, c2, zk)]
        cost_curve += [c2]
        cost_curve_mix += [c1]
        c0, weights = c2, m.export_weights()

    # Display results
    plot_result_lista(cost_ista, cost_fista, cost_curve, cost_curve_mix,
                      _layers, name)


def ensure_plt():
    """Ensure that matplotlib will not fail"""
    r = os.system('python -c "import matplotlib.pyplot as plt;plt.figure()"')
    return r == 0


def plot_result_lista(cost_ista, cost_fista, cost_curve, cost_curve_mix,
                      _layers, name):
    """Plot the curves resulting from our computations
    """
    eps = 1e-6
    c_max = cost_fista[0]
    c_min = min(np.min(cost_fista), np.min(cost_ista), np.min(cost_curve))
    scale = lambda c: (c - c_min) / (c_max - c_min) + eps

    if not ensure_plt():
        import IPython
        IPython.embed()
        mpl.use('Agg')

    import matplotlib.pyplot as plt

    plt.figure(name)
    plt.loglog(range(1, len(cost_ista) + 1), scale(cost_ista), "b",
               label="ISTA")
    plt.loglog(range(1, len(cost_ista) + 1), scale(cost_fista), "g",
               label="FISTA")
    plt.loglog(np.r_[0, _layers], scale(cost_curve), "r", label="LISTA")
    plt.loglog(np.r_[0, _layers], scale(cost_curve_mix), "c", label="LISTAm")
    plt.hlines(eps, 1, _layers[-1] + 10, 'k', '--')
    plt.legend()
    exp_name = f"img/{name}_{datetime.now().strftime('%d%b_%Hh%M')}"
    plt.savefig(exp_name + ".pdf", dpi=150)
    np.save(exp_name, [cost_ista, cost_fista, cost_curve])
    plt.show()

    import IPython
    IPython.embed()


def classification_lista(X, D, Z, lmbd, N_layers, N_epochs, ratio=1 / 3,
                         name="classification"):
    N, p = X.shape

    # Keep 30% of the data for validation
    N_test = int(N * ratio)
    y = parity_test_labels(Z)

    X_test, X_train = X[:N_test], X[N_test:]
    y_test, y_train = y[:N_test], y[N_test:]

    model = lista_network(p, D, n_layers=N_layers, activation="st",
                          lmbd=lmbd)

    # Baseline
    Zk, _ = ista_network(X_test, D, lmbd, N_layers)
    y_pred = parity_test_labels(Zk)
    print(f"Baseline: {np.mean(y_test == y_pred)}")

    # Train the model, iterating on the data in batches of 32 samples
    history = model.fit(X_train, y_train, epochs=N_epochs, batch_size=256,
                        validation_data=(X_test, y_test), shuffle=True)

    if not ensure_plt():
        import IPython
        IPython.embed()
        mpl.use('Agg')

    import matplotlib.pyplot as plt
    plt.figure(name)
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.hlines(acc, 0, args.epochs, linestyle="--")
    plt.title('model loss')
    plt.ylabel('loss')

    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f"{name}_lay{N_layers}_cost.pdf", dpi=150)
    plt.show()
