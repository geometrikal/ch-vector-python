import chvector.transforms.ops as ops
import chvector.transforms.fft as fft
import numpy as np


def log_gabor_spectrum(shape, wavelength, sigma, passband='band'):
    _,_, r, _ =  fft.fft_mesh(shape)
    ops.dc_to_value(r, 1)
    s = np.exp((-(np.log(r * wavelength)) ** 2) / (2 * np.log(sigma) ** 2))
    ops.dc_to_value(r, 0)
    if passband == 'high':
        s[r >= wavelength] = 1
    elif passband == 'low':
        s[r <= wavelength] = 1
    return s



