import keras
import pytest
import numpy as np
from scipy.signal import convolve2d
from keras.layers.merge import Add

from .utils import FLOAT, _soft_thresholding
from .utils import repeat, _test_allclose, _create_conv_data
from .._cost import ConvL2_z
from ..convolutional import lista_conv_network
from ..ista import ista_conv


@repeat(3)
def test_initial_filters_conv_lista():
    # Test the initialization of the filters in conv_lista_network
    # by computing by different mean the filters
    N, d, p, k, c = 10, 5, 32, 5, 3
    rho = .1
    lmbd = 1

    data, D, zs = _create_conv_data(N=N, d=d, p=p, k=k, c=c, rho=rho,
                                    lmbd=lmbd)
    z0 = np.random.normal(size=zs.shape).astype(FLOAT)
    f_cost = ConvL2_z(data, D)

    m = lista_conv_network((c, p, p), D, n_layers=2,
                           activation="st", lmbd=lmbd)

    Wz, Wx = m.Wz, m.Wx

    z = keras.layers.Input(shape=zs.shape[1:])
    x = keras.layers.Input(shape=data.shape[1:])
    gx0_m = m.layers[2](m.layers[1](x))
    gx1_m = m.layers[4](x)
    gz1_m = m.layers[3](z)
    g1_m = Add()([gz1_m, gx1_m])

    m_test = keras.models.Model(
        inputs=[x, z],
        outputs=[gx0_m, g1_m, gx1_m, gz1_m])

    z_l1, z_l2, gx1, gz1 = m_test.predict([data, z0])

    # Compute the output using the convolution to verify
    gz_test = np.array([[[convolve2d(zkk, wzlk, mode='same')
                          for wzlk, zkk in zip(wzl, zn)]
                         for wzl in Wz] for zn in z0]
                       ).sum(axis=2)
    gx_test = np.array([[[convolve2d(xnc, wkc, mode='valid')
                          for wkc, xnc in zip(wk, xn)]
                         for wk in Wx] for xn in data]
                       ).sum(axis=2)
    z_test = gz_test + gx_test
    rec0 = np.array([[[convolve2d(zk, dkc, mode='full')
                       for dkc in dk]
                      for zk, dk in zip(zn, D)] for zn in z0]
                    ).sum(axis=1)
    dz_conv = np.array([[[convolve2d(dkc, rnc, mode='valid')
                          for dkc, rnc in zip(dk, rn)]
                         for dk in D[:, :, ::-1, ::-1]] for rn in rec0]
                       ).mean(axis=2)
    gz_conv = z0 - dz_conv / f_cost.L
    gx_test_st = _soft_thresholding(gx_test, lmbd / f_cost.L)

    z_cost = z0 - f_cost.grad(z0) / f_cost.L
    gx_cost = -f_cost.grad(0 * z0) / f_cost.L
    gx_cost_st = _soft_thresholding(gx_cost, lmbd / f_cost.L)

    # verify cost and convolutions
    _test_allclose(gz_conv + gx_test, z_cost, msg="Wrong filter design for Wz")

    # Ensure that the convolutional layers compute the right convolution
    assert gz1.dtype == gz_test.dtype
    _test_allclose(gz1, gz_test, msg="Convolution mismatched for Wz in Conv2D")
    assert gx1.dtype == gx_test.dtype
    _test_allclose(gx1, gx_test, msg="Convolution mismatched for Wx in Conv2D")

    _test_allclose(gx_cost, gx_test, msg="Wrong filter design for Wx")
    _test_allclose(gx_cost_st, gx_test_st, msg="Issue with soft-thresholding")
    _test_allclose(gx_cost_st, z_l1, msg="Layer 1 mismatched")

    _test_allclose(gz_conv, gz1, msg="Wrong filter design for Wz")
    _test_allclose(gz_conv, gz_test, msg="Wrong filter design for Wz")
    _test_allclose(
        _soft_thresholding(z_cost, lmbd / f_cost.L),
        _soft_thresholding(z_test, lmbd / f_cost.L),
        atol=5e-6
    )
    _test_allclose(z_cost, z_test, msg="Filter design issue", atol=5e-6)
    _test_allclose(z_test, z_l2, msg="Mismatch with Conv2D", atol=5e-6)


@pytest.mark.parametrize("N_layers", [1, 2, 3, 5, 10, 50])
@repeat(4)
def test_cost_conv_lista(N_layers):
    # Test the cost computation using the LISTA model
    N, d, p, k, c = 10, 5, 32, 5, 3
    rho = .02
    lmbd = 1

    data, D, z = _create_conv_data(N=N, d=d, p=p, k=k, c=c, rho=rho, lmbd=lmbd)

    # Base line
    m = lista_conv_network((c, p, p), D, lmbd=lmbd, n_layers=N_layers,
                           activation="st")

    zk, cost_ista = ista_conv(X=data, D=D, lmbd=lmbd, max_iter=N_layers)
    zk = zk[:, :, 0]
    zk2, c_model = m.predict(data, batch_size=len(data))
    c_model = np.mean(c_model)

    # Test cost
    f_cost = ConvL2_z(data, D)

    def _cost_l1(z):
        return lmbd * abs(z).mean(axis=0).sum()

    def _cost(z):
        return f_cost(z) + _cost_l1(z)

    print(f"Cost: ista {_cost(zk)}, lista {_cost(zk2)}, model: {c_model}")
    assert np.isclose(_cost(zk), _cost(zk2))
    assert np.isclose(c_model, _cost(zk2))
