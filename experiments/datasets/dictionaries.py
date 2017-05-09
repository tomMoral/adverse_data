import numpy as np


def create_adversarial_dictionary(K, p, sigma=0, seed=None):

    np.random.seed(seed)

    II = np.random.permutation(K // 2 - 1)
    vec = np.zeros(K)
    D = []

    for k in range(p):
        vec = 0 * vec
        vec[II[k] + 1] = 1 + sigma * np.random.rand()
        D += [np.real(np.fft.ifft(vec))]
    D = np.array(D).T
    D /= np.sqrt((D * D).sum(axis=1, keepdims=True))
    D = D.astype(np.float32)
    return D


def create_gaussian_dictionary(K, p, seed=None):

    np.random.seed(seed)

    D = np.random.normal(size=(K, p))
    D /= np.sqrt((D * D).sum(axis=1))[:, None]
    D = D.astype(np.float32)
    return D


def create_gaussian_conv_dictionary(K, shape, seed=None):

    np.random.seed(seed)

    D = np.random.normal(size=(K,) + shape)
    sum_axis = tuple(range(1, D.ndim))
    D /= np.sqrt((D * D).sum(axis=sum_axis, keepdims=True))
    # D = D.astype(np.float32)
    return D
