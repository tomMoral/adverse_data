import numpy as np
from scipy.signal import convolve2d

from .utils import FLOAT
from .utils import repeat, _test_allclose, _create_conv_data
from .._cost import ConvL2_z


@repeat(5)
def test_ConvL2_z():
    # Test the cost computation using the LISTA model
    N, d, p, k, c = 100, 5, 32, 5, 3
    rho = .02
    lmbd = 1
    eps = 1e-4

    data, D, zs = _create_conv_data(N=N, d=d, p=p, k=k, c=c, rho=rho, lmbd=lmbd)
    f_cost = ConvL2_z(data, D)
    z = np.random.normal(size=zs.shape).astype(FLOAT)

    # Test reconstruct
    rec = f_cost.reconstruct(z, D)
    rec2 = np.array([[[convolve2d(znk, dkc, mode='full') for dkc in dk]
                      for dk, znk in zip(D, zn)]
                     for zn in z]).sum(axis=1)
    _test_allclose(rec, rec2)

    # Test cost computation
    err = rec - data
    _cost = np.mean(err * err, axis=(0, 1)).sum() / 2
    assert np.isclose(f_cost(z), _cost)

    # Test gradient computation
    _grad = [[[convolve2d(rk, dk, mode="valid")
               for rk, dk in zip(residual, Dk)]
              for Dk in D[:, :, ::-1, ::-1]]
             for residual in err]
    _grad = np.mean(_grad, axis=2)

    _test_allclose(f_cost.grad(z), _grad)
    assert f_cost(z - eps * f_cost.grad(z)) < f_cost(z)
    assert f_cost(zs - eps * f_cost.grad(zs)) <= f_cost(zs)
