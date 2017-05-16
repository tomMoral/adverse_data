import numpy as np
from numpy.fft import rfft2 as fft, irfft2 as ifft


def sparse_code(N, D, rho=.02, sigma=1):
    """Create data from Bernouilli gaussian distribution and dictionary D

    Parameters
    ----------
    N (int): Number of sample to generate
    D (array-like: (K, c)): Dictionary to generate the data
    rho (float: .02): sparsity parameter.
    sigma (float: 1.): variance of the gaussian entries.

    Return
    ------
    z (array-like: (N, K)): sampled sparse code
    X (array-like: (N, c)): sampled data points
    """
    K = D.shape[0]
    z = (np.random.rand(N, K) < rho).astype(np.float)
    z *= sigma * np.random.normal(size=(N, K))
    nz = np.isclose(abs(z).sum(axis=1), 0).sum()
    print(f"Generate {nz} 0 with this sparse code")
    return z, z.dot(D)


def conv_sparse_code(N, w, D, rho=.02, sigma=10):
    """Create data from Bernouilli gaussian distribution and dictionary D

    Parameters
    ----------
    N (int): Number of sample to generate
    w (int): Width of the generated image
    D (array-like: (K, c, k, k)): Dictionary to generate the data
    rho (float: .02): sparsity parameter.
    sigma (float: 1.): variance of the gaussian entries.

    Return
    ------
    z (array-like: (N, K, w-k+1, w-k+1)): sampled sparse code
    X (array-like: (N, c, w, w)): sampled data points
    """
    K, c, k, _ = D.shape

    fft_shape = w + k - 1, w + k - 1

    # generate z support
    z_shape = N, K, w - k + 1, w - k + 1
    print("Generate z support")
    z = (np.random.random(z_shape) < rho).astype(np.float)
    print("Generate z coefficients")
    z *= sigma * np.random.normal(size=z_shape)

    print("Non-Zero  in z:", 1 - np.isclose(z, 0).sum() / z.size)

    z_fft = fft(z, s=fft_shape)
    D_fft = fft(D, s=fft_shape)
    X = ifft((D_fft[None] * z_fft[:, :, None]).sum(axis=1))[:, :, :w, :w]

    return z, X

