import numpy as np
from .. import dictionaries


def test_dictionnary():
    d, c, k = 100, 3, 4
    D = dictionaries.create_gaussian_conv_dictionary(d, (c, k, k))

    assert np.allclose((D * D).sum(axis=(1, 2, 3)), 1)

    d, c = 100, 100
    D = dictionaries.create_gaussian_dictionary(d, c)

    assert np.allclose((D * D).sum(axis=1), 1)
