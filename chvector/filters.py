import chvector.transforms.ops as ops
import numpy as np
from scipy.fftpack import fft2, ifft2, ifftshift


def fft_mesh(shape):
    rows = shape[0]
    cols = shape[1]
    ix = (np.arange(0, cols) - np.fix(cols / 2)) / (cols - np.mod(cols, 2))
    iy = (np.arange(0, rows) - np.fix(rows / 2)) / (rows - np.mod(rows, 2))

    ux, uy = np.meshgrid(ix, iy)
    ux = ifftshift(ux)
    uy = ifftshift(uy)
    if np.ndim(shape) == 3:
        ux = np.repeat(ux[:, :, np.newaxis], shape[3], axis=2)
        uy = np.repeat(uy[:, :, np.newaxis], shape[3], axis=2)
    r = np.sqrt(ux**2 + uy**2)
    th = np.arctan2(uy, ux)

    return ux, uy, r, th


def img_fft2(im):
    return fft2(im, axes=[0, 1])


def img_ifft2(f):
    return ifft2(f, axes=[0, 1])


def log_gabor_spectrum(shape, wavelength, sigma, passband='band'):
    _,_, r, _ =  fft_mesh(shape)
    ops.dc_to_value(r, 1)
    s = np.exp((-(np.log(r * wavelength)) ** 2) / (2 * np.log(sigma) ** 2))
    ops.dc_to_zero(s)
    if passband == 'high':
        s[r >= 1/wavelength] = 1
    elif passband == 'low':
        s[r <= 1/wavelength] = 1
        ops.dc_to_value(s, 1)
    return s


def scale_adaptive_spectrums(shape, cutoff, N):
    _, _, f, _ = fft_mesh(shape)
    w = 2 * np.pi * f  # convert to radians
    w_c = np.pi / cutoff
    r_c = np.log2(w_c / 2)
    r_w = np.log2(np.sqrt(2) * np.pi) - r_c
    r = np.pi * (np.log2(w) - r_c) / r_w
    h_w = np.cos(np.pi/2 * np.log2(w / w_c))
    h_w[w <= w_c/2] = 0
    h_w[w > w_c] = 1
    h = np.zeros(shape + (2*N+1, ))
    for n in range(N+1):
        if n == 0:
            h[:, :, n] = h_w
        else:
            h[:, :, 2 * n - 1] = h_w * np.cos(n * r)
            h[:, :, 2 * n] = h_w * np.sin(n * r)
    h[np.isnan(h)] = 0
    return h


def apply_filter(im, f):
    return img_ifft2(img_fft2(im) * f)


if __name__ == "__main__":
    pass







