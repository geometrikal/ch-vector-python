import numpy as np
import skimage.io as skio
import skimage.transform as skt
import skimage.color as skc
from scipy.stats import beta

from chvector.filters import log_gabor_spectrum
from chvector.transforms.chv import img_chv, chv_norm
import matplotlib.pyplot as plt


def phase_congruency(im, N, w, sigma, scales, scale_factor=2):
    plt.matshow(im)
    plt.show()
    if np.ndim(im) > 2:
        im = skc.rgb2gray(im)
    ch = np.zeros(im.shape + (2*N+1, scales), dtype=np.complex)
    for i in range(scales):
        spectrum = log_gabor_spectrum(im.shape[:2], w * scale_factor**i, sigma, passband='band')
        ch[:, :, :, i] = img_chv(im, spectrum, N)
        plt.matshow(chv_norm(ch[:, :, :, i]))
        plt.show()
    A1 = np.linalg.norm(np.sum(ch, axis=-1), axis=-1)
    A2 = np.sum(np.linalg.norm(ch, axis=-2).squeeze(), axis=-1)
    plt.matshow(A1)
    plt.colorbar()
    plt.show()
    plt.matshow(A2)
    plt.colorbar()
    plt.show()

    epsilon = 0.01
    plt.matshow(A1 / (A2 + epsilon))
    plt.colorbar()
    plt.show()

    plt.matshow(beta.cdf(A1 / (A2 + epsilon), 5, 1))
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    im = skio.imread("test/coral.jpg")
    im = skt.rescale(im, [0.25, 0.25, 1])
    phase_congruency(im,
                     N=3,
                     w=4,
                     sigma=0.65,
                     scales=4,
                     scale_factor=2)