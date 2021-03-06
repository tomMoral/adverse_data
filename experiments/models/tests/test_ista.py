import numpy as np

from .utils import FLOAT, _soft_thresholding
from .utils import repeat, _test_allclose, _create_conv_data, _create_data
from .._cost import ConvL2_z
from ..ista import ista, fista
from ..ista_network import ista_network, fista_network
from ..ista import ista_conv, fista_conv
from ..ista_network import ista_conv_network, fista_conv_network


@repeat(3)
def test_ista():
    # Test the initialization of the filters in conv_lista_network
    # by computing by different mean the filters
    N, d, p = 10, 5, 32
    rho = .1
    lmbd = 1

    data, D, zs = _create_data(N=N, d=d, p=p, rho=rho, lmbd=lmbd)
    L = np.linalg.norm(D, ord=2)**2

    # Define the cost function
    def _cost(zk):
        err = data - zk.dot(D)
        l2 = np.sum(err * err, axis=1).mean()
        l1 = np.sum(abs(zk), axis=1).mean()
        return 1 / 2 * l2 + lmbd * l1

    for _ in range(10):
        z0 = np.random.normal(size=zs.shape).astype(FLOAT)
        z1, cost = ista(data, D, lmbd, max_iter=1, z0=z0)
        assert cost[0] > cost[1]
        assert np.isclose(_cost(z1), cost[1])

    # Test with a 0 gradient solution to verify the prox
    z1, cost = ista(data, D, lmbd, max_iter=1, z0=zs)
    assert cost[0] > cost[1]
    assert np.isclose(_cost(z1), cost[1])
    _test_allclose(z1, _soft_thresholding(zs, lmbd / L))


@repeat(3)
def test_ista_network():
    # Test the initialization of the filters in conv_lista_network
    # by computing by different mean the filters
    N, d, p = 10, 5, 32
    rho = .1
    lmbd = 1

    data, D, zs = _create_data(N=N, d=d, p=p, rho=rho, lmbd=lmbd)
    L = np.linalg.norm(D, ord=2)**2

    # Define the cost function
    def _cost(zk):
        err = data - zk.dot(D)
        l2 = np.sum(err * err, axis=1).mean()
        l1 = np.sum(abs(zk), axis=1).mean()
        return 1 / 2 * l2 + lmbd * l1

    for _ in range(10):
        z0 = np.random.normal(size=zs.shape).astype(FLOAT)
        z1, cost = ista_network(data, D, lmbd, max_iter=1, z0=z0)
        assert cost[0] > cost[1]
        assert np.isclose(_cost(z1), cost[1])

    # Test with a 0 gradient solution to verify the prox
    z1, cost = ista(data, D, lmbd, max_iter=1, z0=zs)
    assert cost[0] > cost[1]
    assert np.isclose(_cost(z1), cost[1])
    _test_allclose(z1, _soft_thresholding(zs, lmbd / L))


@repeat(3)
def test_ista_conv():
    # Test the initialization of the filters in conv_lista_network
    # by computing by different mean the filters
    N, d, p, k, c = 10, 5, 32, 5, 3
    rho = .1
    lmbd = 1

    data, D, zs = _create_conv_data(N=N, d=d, p=p, k=k, c=c, rho=rho,
                                    lmbd=lmbd)

    # Define the cost function
    f_cost = ConvL2_z(data, D)

    def _cost(z):
        return f_cost(z) + lmbd * abs(z).mean(axis=0).sum()

    for _ in range(10):
        z0 = np.random.normal(size=zs.shape).astype(FLOAT)
        z1, cost = ista_conv(data, D, lmbd, max_iter=1, z0=z0)
        assert cost[0] > cost[1]
        assert np.isclose(_cost(z1), cost[1])

    # Test with a 0 gradient solution to verify the prox
    z1, cost = ista_conv(data, D, lmbd, max_iter=1, z0=zs)
    assert cost[0] > cost[1]
    assert np.isclose(_cost(z1), cost[1])
    _test_allclose(z1, _soft_thresholding(zs, lmbd / f_cost.L))


@repeat(3)
def test_ista_conv_network():
    # Test the initialization of the filters in conv_lista_network
    # by computing by different mean the filters
    N, d, p, k, c = 10, 5, 32, 5, 3
    rho = .1
    lmbd = 1

    data, D, zs = _create_conv_data(N=N, d=d, p=p, k=k, c=c, rho=rho,
                                    lmbd=lmbd)

    # Define the cost function
    f_cost = ConvL2_z(data, D)

    def _cost(z):
        return f_cost(z) + lmbd * abs(z).mean(axis=0).sum()

    for _ in range(10):
        z0 = np.random.normal(size=zs.shape).astype(FLOAT)
        z1, cost = ista_conv_network(data, D, lmbd, max_iter=1, z0=z0)
        assert cost[0] > cost[1]
        assert np.isclose(_cost(z1), cost[1])

    # Test with a 0 gradient solution to verify the prox
    z1, cost = ista_conv_network(data, D, lmbd, max_iter=1, z0=zs)
    assert cost[0] > cost[1]
    assert np.isclose(_cost(z1), cost[1])
    _test_allclose(z1, _soft_thresholding(zs, lmbd / f_cost.L))


@repeat(3)
def test_fista():
    # Test the initialization of the filters in conv_lista_network
    # by computing by different mean the filters
    N, d, p = 10, 5, 32
    rho = .1
    lmbd = 1

    data, D, zs = _create_data(N=N, d=d, p=p, rho=rho, lmbd=lmbd)

    for _ in range(10):
        z0 = np.random.normal(size=zs.shape).astype(FLOAT)
        z1, cost = ista(data, D, lmbd, max_iter=1, z0=z0)
        z2, cost = fista(data, D, lmbd, max_iter=1, z0=z0)

        _test_allclose(z1, z2)


@repeat(3)
def test_fista_network():
    # Test the initialization of the filters in conv_lista_network
    # by computing by different mean the filters
    N, d, p = 10, 5, 32
    rho = .1
    lmbd = 1

    data, D, zs = _create_data(N=N, d=d, p=p, rho=rho, lmbd=lmbd)

    for _ in range(10):
        z0 = np.random.normal(size=zs.shape).astype(FLOAT)
        z1, cost = ista_network(data, D, lmbd, max_iter=1, z0=z0)
        z2, cost = fista_network(data, D, lmbd, max_iter=1, z0=z0)

        _test_allclose(z1, z2)


@repeat(3)
def test_fista_conv():
    # Test the initialization of the filters in conv_lista_network
    # by computing by different mean the filters
    N, d, p, k, c = 10, 5, 32, 5, 3
    rho = .1
    lmbd = 1

    data, D, zs = _create_conv_data(N=N, d=d, p=p, k=k, c=c, rho=rho,
                                    lmbd=lmbd)

    for _ in range(10):
        z0 = np.random.normal(size=zs.shape).astype(FLOAT)
        z1, cost = ista_conv(data, D, lmbd, max_iter=1, z0=z0)
        z2, cost = fista_conv(data, D, lmbd, max_iter=1, z0=z0)

        _test_allclose(z1, z2)


@repeat(3)
def test_fista_conv_network():
    # Test the initialization of the filters in conv_lista_network
    # by computing by different mean the filters
    N, d, p, k, c = 10, 5, 32, 5, 3
    rho = .1
    lmbd = 1

    data, D, zs = _create_conv_data(N=N, d=d, p=p, k=k, c=c, rho=rho,
                                    lmbd=lmbd)

    for _ in range(10):
        z0 = np.random.normal(size=zs.shape).astype(FLOAT)
        z1, cost = ista_conv_network(data, D, lmbd, max_iter=1, z0=z0)
        z2, cost = fista_conv_network(data, D, lmbd, max_iter=1, z0=z0)

        _test_allclose(z1, z2)
