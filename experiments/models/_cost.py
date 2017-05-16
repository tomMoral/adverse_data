import numpy as np
from numpy import linalg as LA
from scipy import fftpack
from numpy.fft import rfft2 as fft, irfft2 as ifft


class LossL2(object):
    def __init__(self, X, D):
        self.X = X
        self.D = D
        self._compute_constant()

    def _compute_constant(self):
        # Lipchitz constant
        self.DtD = self.D.dot(self.D.T)
        self.DtX = self.X.dot(self.D.T)
        self.L = np.linalg.norm(self.DtD, ord=2)

        # Store extra dimensions
        self.K, self.p = self.D.shape
        self.N = np.prod(self.X.shape[:-1])
        self.z_shape = (self.N, self.K)

    def get_z0(self):
        return np.zeros(self.z_shape)

    def __call__(self, z):
        """Compute the value of the Cost function"""
        assert z.shape[:-1] == self.X.shape[:-1]
        aux = self.X - z.dot(self.D)
        return np.sum(aux * aux) / 2

    def grad(self, z, out=None):
        """Compute the gradient of the cost at point z.

        Parameters
        ----------
        z : array-like
            current point for the gradient computation
        out: array like (default: None)
            output array, if it is not None, out should have the same size as
            z and will be use to store the result.
        """
        if out is None:
            out = np.empty(z.shape)
        out[:] = (z.dot(self.DtD) - self.DtX)
        return out

    def zstep(self, t, z_dual, mu, out=None):
        """Compute the ADMM step associated to this component of the cost.

        Parameters
        ----------
        t, z_dual : array-like
            current split variable and dual vaariable for z,
        mu : float
            current multiplier value for z,
        out: array like (default: None)
            output array, if it is not None, out should have the same size as
            z and will be use to store the result.
        """

        A = mu * np.eye(self.K) + self.DtD
        b = (self.Dtx_fft + mu * t - z_dual).T

        if out is None:
            out = np.empty(t.shape)
        out[:] = LA.solve(A, b).T
        return out

    @classmethod
    def reconstruct(cls, z, D):
        return np.sum(D[None] * z[:, :, None], axis=1)


class ConvL2_z(object):
    def __init__(self, X, D):
        self.X = X
        self.D = D

        self._compute_constant()

    def _compute_constant(self):
        """Precompute fft of X and D to fasten the gradient computations"""
        N, p = self.X.shape[:2]
        X_shape = self.X.shape[-2:]
        z_shape = np.array(X_shape) - np.array(self.D.shape[-2:]) + 1
        fft_shape = np.array(X_shape) + np.array(self.D.shape[-2:]) - 1

        # Frequential domain representation
        self.fft_shape = fft_shape = [fftpack.helper.next_fast_len(int(d))
                                      for d in fft_shape]
        self.X_slice = [slice(0, d) for d in self.X.shape]
        self.X_fft = X_fft = fft(self.X, s=fft_shape)
        self.D_fft = D_fft = fft(self.D, s=fft_shape)

        # Reshape so that all the variableas have the same dimensions
        # [N, K, p, w, h]
        self.X_fft = X_fft = self.X_fft[:, None]
        # optimizing the code
        if self.D.ndim == 4:
            self.D = self.D[None]
            self.D_fft = D_fft = self.D_fft[None]
        K = self.D_fft.shape[1]
        self.z_fft_shape = (N, K, 1) + tuple(fft_shape)
        self.z_shape = (N, K, 1) + tuple(z_shape)

        # Precompute constants to accelerate frequency domain computations
        self.DtD_fft = (D_fft[:, :, None].conj() * D_fft[:, None]
                        ).mean(axis=3, keepdims=True)
        self.DtX_fft = (D_fft.conj() * X_fft).mean(axis=2, keepdims=True)

        # Lipchitz constant
        self.L = LA.norm(self.DtD_fft, axis=(1, 2), ord=2).max()

        # Store extra dimensions
        self.T = N * np.prod(X_shape)

    def get_z0(self):
        return np.zeros(self.z_shape)

    def __call__(self, z):
        """Compute the value of the Cost function"""
        if z.ndim == 4:
            z = z[:, :, None]
        z_fft = fft(z, s=self.fft_shape)
        aux = self.X_fft - np.sum(self.D_fft * z_fft, axis=1, keepdims=True)
        diff = ifft(aux)[self.X_slice[:1] + [0] + self.X_slice[1:]]
        return np.mean(diff * diff, axis=(0, 1)).sum() / 2

    def grad(self, z, out=None):
        """Compute the gradient of the cost at point z.

        Parameters
        ----------
        z : array-like
            current point for the gradient computation
        out: array like (default: None)
            output array, if it is not None, out should have the same size as
            z and will be use to store the result.

        The gradient is not scaled with the number of samples N used as this
        scaling is canceled out by the step size N/L.
        """
        if out is None:
            out = np.empty(z.shape, dtype=z.dtype)
        if z.ndim == 4:
            z = z[:, :, None]
            z_slice = [slice(0, d) for d in z.shape]
            z_slice[2] = 0
        else:
            z_slice = [slice(0, d) for d in z.shape]
        z_fft = fft(z, s=self.fft_shape)
        Gh = np.sum(self.DtD_fft * z_fft[:, None], axis=2)
        Gh -= self.DtX_fft
        out[:] = ifft(Gh).real[z_slice]
        return out

    def zstep(self, t, z_dual, mu, out=None):
        """Compute the ADMM step associated to this component of the cost.

        Parameters
        ----------
        t, z_dual : array-like
            current split variable and dual vaariable for z,
        mu : float
            current multiplier value for z,
        out: array like (default: None)
            output array, if it is not None, out should have the same size as
            z and will be use to store the result.
        """
        z_slice = [slice(0, d) for d in t.shape]
        t_fft = fft(t, s=self.fft_shape)
        z_dual_fft = fft(z_dual, s=self.fft_shape)

        A = mu * np.eye(self.K) + self.DtD_fft.swapaxes(0, 1).T
        b = (self.Dtx_fft + mu * t_fft - z_dual_fft).T
        z_fft = LA.solve(A, b).T

        if out is None:
            out = np.empty(t.shape)
        out[:] = ifft(z_fft.reshape(self.z_fft_shape))[z_slice]
        return out

    @classmethod
    def reconstruct(cls, z, D):
        X_shape = np.array(z.shape[-2:]) + np.array(D.shape[-2:]) - [1, 1]
        fft_shape = [fftpack.helper.next_fast_len(int(d))
                     for d in X_shape[-2:]]
        X_shape = (z.shape[0], D.shape[1], ) + tuple(X_shape)
        X_slice = [slice(0, d) for d in X_shape]
        D_fft = fft(D, s=fft_shape)
        z_fft = fft(z, s=fft_shape)
        if z_fft.ndim == 4:
            z_fft = z_fft[:, :, None]
        if D_fft.ndim == 4:
            D_fft = D_fft[None]
        rec = (D_fft * z_fft).sum(axis=1)
        return ifft(rec)[X_slice]


class ConvL2_D(object):
    def __init__(self, X, Z):
        self.X = X
        self.Z = Z

        self._compute_constant()

    def _compute_constant(self):
        """Precompute fft of X and D to fasten the gradient computations"""
        N, p = self.X.shape[:2]
        X_shape = self.X.shape[-2:]
        D_shape = np.array(X_shape) - np.array(self.Z.shape[-2:]) + 1
        fft_shape = np.array(X_shape) + np.array(D_shape) - 1

        # Frequential domain representation
        self.fft_shape = fft_shape = [fftpack.helper.next_fast_len(int(d))
                                      for d in fft_shape]
        self.X_slice = [slice(0, d) for d in self.X.shape]
        self.X_fft = X_fft = fft(self.X, s=fft_shape)
        self.Z_fft = Z_fft = fft(self.Z, s=fft_shape)

        # Reshape so that all the variableas have the same dimensions
        # [N, K, p, w, h]
        self.X_fft = X_fft = self.X_fft[:, None]
        # Then we are optimizing the dictionary
        if self.Z_fft.ndim == 4:
            self.Z_fft = Z_fft = self.Z_fft[:, :, None]
        K = self.Z.shape[1]
        self.D_fft_shape = (1, K, p) + tuple(fft_shape)
        self.D_shape = (1, K, p) + tuple(D_shape)

        # Precompute constants to accelerate frequency domain computations
        self.ZtZ_fft = (Z_fft[:, :, None].conj() * Z_fft[:, None]
                        ).mean(axis=0, keepdims=True)
        self.ZtX_fft = (Z_fft.conj() * X_fft).mean(axis=0, keepdims=True)

        # Lipchitz constant
        self.L = np.linalg.norm(self.ZtZ_fft, axis=(1, 2), ord=2).max()

        # Store extra dimensions
        self.T = N * np.prod(X_shape)

    def get_z0(self):
        return np.zeros(self.D_shape)

    def __call__(self, D):
        """Compute the value of the Cost function"""
        if D.ndim == 4:
            D = D[None]
        D_fft = fft(D, s=self.fft_shape)
        aux = self.X_fft - np.sum(self.Z_fft * D_fft, axis=1, keepdims=True)
        return np.sum(aux.conj() * aux).real / (2 * self.T)

    def grad(self, D, out=None):
        """Compute the gradient of the cost at point z.

        Parameters
        ----------
        D : array-like
            current point for the gradient computation
        out: array like (default: None)
            output array, if it is not None, out should have the same size as
            z and will be use to store the result.

        The gradient is not scaled with the number of samples N used as this
        scaling is canceled out by the step size N/L.
        """
        D_shape = D.shape
        if D.ndim == 4:
            D = D[None]
            D_slice = [slice(0, d) for d in D.shape]
            D_slice[0] = 0
        else:
            D_slice = [slice(0, d) for d in D.shape]
        D_fft = fft(D, s=self.fft_shape)
        Gh = np.sum(self.ZtZ_fft * D_fft[:, None], axis=2)
        Gh -= self.ZtX_fft
        if out is None:
            out = np.empty(D_shape)
        out[:] = ifft(Gh).real[D_slice]
        return out

    def zstep(self, t, z_dual, mu, out=None):
        """Compute the ADMM step associated to this component of the cost.

        Parameters
        ----------
        t, z_dual : array-like
            current split variable and dual vaariable for z,
        mu : float
            current multiplier value for z,
        out: array like (default: None)
            output array, if it is not None, out should have the same size as
            z and will be use to store the result.
        """
        z_slice = [slice(0, d) for d in t.shape]
        t_fft = fft(t, s=self.fft_shape)
        z_dual_fft = fft(z_dual, s=self.fft_shape)

        A = mu * np.eye(self.K) + self.DtD_fft.swapaxes(0, 1).T
        b = (self.Dtx_fft + mu * t_fft - z_dual_fft).T
        z_fft = LA.solve(A, b).T

        if out is None:
            out = np.empty(t.shape)
        out[:] = ifft(z_fft.reshape(self.z_fft_shape))[z_slice]
        return out


class L1Reg(object):
    def __init__(self, lmbd, positive=False):
        self.lmbd = lmbd
        self.positive = positive

    def __call__(self, z):
        """Compute the value of the cost function at point z"""
        # average over the number of sample to have a regularzation scaling
        # lmbd invariant to this quantity.
        N = z.shape[0]
        return self.lmbd * np.sum(abs(z)) / N

    def prox(self, z, mu, out=None):
        """Compute the proximal operator of the cost at point z.

        Parameters
        ----------
        z : array-like
            current point for the prox computation
        out: array like (default: None)
            output array, if it is not None, out should have the same size as
            z and will be use to store the result.
        """
        N = z.shape[0]
        if self.positive:
            return np.maximum(z - self.lmbd * mu / N, 0, out)

        if out is None:
            out = np.empty(z.shape)
        out[:] = np.sign(z) * np.clip(abs(z) - self.lmbd * mu / N, 0, np.inf)
        return out


class cL1Reg(object):
    def __init__(self, lmbd, positive=False):
        self.lmbd = lmbd
        self.positive = positive

    def __call__(self, z):
        """Compute the value of the cost function at point z"""
        # average over the number of samples and the length of the serie to
        # have a regularzation scaling lmbd invariant to these quantities.
        N = z.shape[0]
        T = np.prod(z.shape[-2:]) * N
        return self.lmbd * np.sum(abs(z)) / T

    def prox(self, z, mu, out=None):
        """Compute the proximal operator of the cost at point z.

        Parameters
        ----------
        z : array-like
            current point for the prox computation
        out: array like (default: None)
            output array, if it is not None, out should have the same size as
            z and will be use to store the result.
        """
        # average with the length of the serie only. The average on the number
        # of sample canceled out with the step size.
        T = np.prod(z.shape[-2:])
        if self.positive:
            return np.maximum(z - self.lmbd * mu / T, 0, out)

        if out is None:
            out = np.empty(z.shape)
        out[:] = np.sign(z) * np.clip(abs(z) - self.lmbd * mu / T, 0, np.inf)
        return out

    def zstep(self, t, z_dual, mu_z, step_size, out=None):
        """Compute the ADMM step associated to this component of the cost.

        Parameters
        ----------
        t, z_dual : array-like
            current split variable and dual vaariable for z,
        mu_z, step_size : float
            current multiplier value and step_size for z,
        out: array like (default: None)
            output array, if it is not None, out should have the same size as
            z and will be use to store the result.
        """
        if out is None:
            out = np.empty(t.shape)
        if mu_z > 0:
            out[:] = z_dual / mu_z + t
            self._prox(out, mu=step_size / mu_z, out=out)
        else:
            out[:] = 0
        return out


class L2Reg(object):
    def __init__(self, lmbd):
        self.lmbd = lmbd
        self.L = self.lmbd

    def __call__(self, z):
        """Compute the value of the cost function at point z"""
        # average over the number of samples and the length of the serie to
        # have a regularzation scaling lmbd invariant to these quantities.
        return self.lmbd * np.sum(z * z) / 2

    def grad(self, z, out=None):
        """Compute the proximal operator of the cost at point z.

        Parameters
        ----------
        z : array-like
            current point for the prox computation
        out: array like (default: None)
            output array, if it is not None, out should have the same size as
            z and will be use to store the result.
        """
        if out is None:
            out = np.empty(z.shape)
        out[:] = self.lmbd * z
        return out


class ElasticNet(object):
    """ElasticNet regularization
    """
    def __init__(self, lmbd, beta=0, positive=False):
        self.lmbd = lmbd
        self.beta = beta
        self.positive = positive

    def __call__(self, z):
        """Compute the value of the cost function at point z"""
        # average over the number of samples and the length of the serie to
        # have a regularzation scaling lmbd invariant to these quantities.
        N = z.shape[0]
        return (self.lmbd * np.sum(abs(z)) + self.beta * np.sum(z * z)) / N

    def prox(self, z, mu, out=None):
        """Compute the proximal operator of the cost at point z.

        Parameters
        ----------
        z : array-like
            current point for the prox computation
        out: array like (default: None)
            output array, if it is not None, out should have the same size as
            z and will be use to store the result.
        """
        N = z.shape[0]
        beta = self.beta
        if self.positive:
            return np.maximum(z - mu * self.lmbd / N, 0, out) \
                / (1 + 2 * beta / N)

        if out is None:
            out = np.empty(z.shape)
        out[:] = np.sign(z) * np.clip(abs(z) - mu * self.lmbd / N, 0, np.inf) \
            / (1 + 2 * mu * beta / N)
        return out


class Cost(object):
    def __init__(self, components):
        self.components = components
        self.smooth_components = []
        self.nonsmooth_components = []
        self.L = 0
        for component in components:
            if hasattr(component, "grad"):
                self.smooth_components += [component]
                self.L += component.L
            else:
                self.nonsmooth_components += [component]

    def grad(self, z, out=None):
        if len(self.smooth_components) == 1:
            return self.smooth_components[0].grad(z, out)
        if out is None:
            out = np.zeros(z.shape)
        else:
            out[:] = 0

        for component in self.smooth_components:
            out += component.grad(z)

        return out

    def prox(self, z, mu, out=None):
        """Return a prox for all nonsmooth components in this cost.

        The proximal operator is computed by chaining the proximal operators
        of each components. This methods is suppose to converge for proper
        convex functions. (See combettes and al, 07)
        """
        if len(self.nonsmooth_components) == 0:
            return z
        if len(self.nonsmooth_components) == 1:
            return self.nonsmooth_components[0].prox(z, mu, out=out)

        if out is None:
            out = self.nonsmooth_components[0].prox(z, mu)
        else:
            self.nonsmooth_components[0].prox(z, mu, out=out)

        for component in self.smooth_components[1:]:
            component.prox(out, mu, out)
        return out

    def __call__(self, z):
        cost = 0
        for component in self.components:
            cost += component(z)
        return cost

    def reg(self, z):
        cost = 0
        for component in self.nonsmooth_components:
            cost += component(z)
        return cost

    def get_z0(self):
        return self.smooth_components[0].get_z0()
