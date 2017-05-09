import numpy as np
from numpy.fft import rfft2 as fft, irfft2 as ifft

from ...datasets.dictionaries import create_gaussian_conv_dictionary


# If set to true, _test_allclose display the discrepency between a and b
DISPLAY = True

# Type of data created
FLOAT = np.float32


def repeat(n_rep):
    """Decorator to repeat a test multiple times
    """
    def decorator(func):
        func._repeat = n_rep
        return func
    return decorator


def _soft_thresholding(x, mu):
    return np.sign(x) * np.maximum(abs(x) - mu, 0)


def _test_allclose(a, b, msg="a, b", atol=1e-6):
    """Raise an exception if a and b are not equals.

    This function will also show the difference between the 2 in a plot if
    DISPLAY is set.
    """
    L_inf = abs(a - b).max()
    L_1 = abs(a - b).sum()
    L_0 = 1 - np.isclose(a, b).sum() / a.size

    try:
        assert np.allclose(a, b, atol=atol), (
            f"NotEqual: {msg}:\nL_0: {L_0}\nL_1: {L_1}\nL_inf: {L_inf}")
    except AssertionError:
        if DISPLAY:
            import matplotlib.pyplot as plt
            plt.imshow((1 - np.isclose(a, b)).sum(axis=(0, 1)))
            plt.show()
        raise


def _create_conv_data(N=10, d=1, p=64, k=3, c=3, rho=.02, lmbd=1):

    fft_shape = p + k - 1, p + k - 1
    D = create_gaussian_conv_dictionary(d, (c, k, k))
    D_fft = fft(D, s=fft_shape)

    w = h = p
    # generate z support
    print("Generate z support")
    z = (np.random.rand(N, d, w - k + 1, h - k + 1) < rho).astype(FLOAT)
    print("Generate z coefficients")
    z *= 10 * np.random.normal(size=(N, d, w - k + 1, h - k + 1))

    print("Non-Zero  in z:", 1 - np.isclose(z, 0).sum() / z.size)

    z_fft = fft(z, s=fft_shape)
    data = ifft((D_fft[None] * z_fft[:, :, None]).sum(axis=1))[:, :, :p, :p]

    return data.astype(FLOAT), D.astype(FLOAT), z.astype(FLOAT)
