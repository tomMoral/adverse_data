import numpy as np


def sign_test_labels(z):
    """Either positive ornegative coefficients hold more power
    """

    return np.array([zz[zz > 0].sum() > - zz[zz < 0].sum() for zz in z]
                    ).reshape((-1, 1))


def parity_test_labels(z):
    """Either odd or even coefficients hold more power
    """

    return np.array([abs(zz[::2]).sum() > abs(zz[1::2]).sum() for zz in z]
                    ).reshape((-1, 1))
