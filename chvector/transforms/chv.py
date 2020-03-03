import numpy as np
import chvector.transforms.fft as fft
import chvector.transforms.ops as ops


def img_chv(im, basis_filter_spectrum, N, weights=None):
    assert np.ndim(im) == 2, "Image should only have 2 dimensions (greyscale)"

    if weights is None:
        weights = np.ones(2*N+1)

    # FFT of image
    f = fft.img_fft2(im)
    f *= basis_filter_spectrum

    # CHV placeholder
    ch = np.zeros(f.shape + (2*N+1,), dtype=np.complex)

    # Calculate each order
    for n in range(-N, N+1):
        t = rt_spectrum(f.shape, n)
        ch[:, :, n + N] = fft.img_ifft2(f * t) * weights[n + N]
    return ch


def rt_spectrum(shape, n):

    if n == 0:
        return np.ones(shape)

    # Get spectrum coordinates
    ux, uy, _, _ = fft.fft_mesh(shape)

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