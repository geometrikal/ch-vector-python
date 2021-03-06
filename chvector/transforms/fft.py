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
        ux = np.repeat(ux[:,:,np.newaxis], shape[3], axis=2)
        uy = np.repeat(uy[:,:,np.newaxis], shape[3], axis=2)
    r = np.sqrt(ux**2 + uy**2)
    th = np.arctan2(uy, ux)

    return ux, uy, r, th


def img_fft2(im):
    return fft2(im, axes=[0, 1])


def img_ifft2(f):
    return ifft2(f, axes=[0, 1])


