import numpy as np
import chvector.filters.filters as filt
import chvector.transforms.ops as ops
import chvector.models.weights as chw
from scipy.fftpack import ifftshift


def img_chv(im, basis_filter_spectrum, N, weights='sinusoid'):

    if isinstance(weights, str) and weights == 'sinusoid':
        weights = chw.sinusoid(N)

    if np.ndim(im) > 2 and im.shape[2] > 1:
        r = np.zeros(im.shape + (2*N+1,), dtype=np.complex)
        for i in range(im.shape[2]):
            r[..., i, :] = img_chv(im[..., i], basis_filter_spectrum, N, weights)
        return r

    if weights is None:
        weights = np.ones(2*N+1)

    # FFT of image
    f = filt.img_fft2(im)
    f *= basis_filter_spectrum

    # CHV placeholder
    ch = np.zeros(f.shape + (2*N+1,), dtype=np.complex)

    # Calculate each order
    for n in range(-N, N+1):
        t = rt_spectrum(f.shape, n)
        ch[:, :, n + N] = filt.img_ifft2(f * t) * weights[n + N]
    return ch


def rt_spectrum(shape, n):
    if n == 0:
        return np.ones(shape)

    # Get spectrum coordinates
    ux, uy, _, _ = filt.fft_mesh(shape)

    # Calculate RT spectrum
    if n < 0:
        v = ux - 1j * uy
        n = -n
    else:
        v = ux + 1j * uy
    r = np.abs(v)
    ops.dc_to_value(r, 1)
    v = (v / r) ** n
    ops.dc_to_zero(v)
    return v