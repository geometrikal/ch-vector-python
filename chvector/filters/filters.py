import chvector.transforms.ops as ops
import chvector.transforms.fft as fft
import numpy as np


def log_gabor_spectrum(shape, wavelength, sigma, passband='band'):
    _,_, r, _ =  fft.fft_mesh(shape)
    ops.dc_to_value(r, 1)
    s = np.exp((-(np.log(r * wavelength)) ** 2) / (2 * np.log(sigma) ** 2))
    if passband == 'high':
        s[r >= 1/wavelength] = 1
    elif passband == 'low':
        s[r <= 1/wavelength] = 1
        ops.dc_to_value(s, 1)
    return s


def apply_filter(im, f):
    return fft.img_ifft2(fft.img_fft2(im) * f)



