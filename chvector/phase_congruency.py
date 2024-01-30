import numpy as np
import skimage.io as skio
import skimage.transform as skt
import skimage.color as skc
from scipy.stats import beta

from chvector.filters import log_gabor_spectrum
from chvector.models.archetypes import sinusoid_pair
from chvector.models.solver import create_poly
from chvector.transforms.chv import img_chv, chv_norm
import matplotlib.pyplot as plt


def phase_congruency(im, N, w, sigma, scales, scale_factor=2, model=None, model_factor=1):
    plt.matshow(im)
    plt.show()
    if np.ndim(im) > 2:
        im = skc.rgb2gray(im)
    ch = np.zeros(im.shape + (2*N+1, scales), dtype=complex)

    for i in range(scales):
        spectrum = log_gabor_spectrum(im.shape[:2], w * scale_factor**i, sigma, passband='band')
        ch_i = img_chv(im, spectrum, N)
        if model is not None:
            _, ch_i, _ = create_poly(ch_i, model)
            # print(ch_i[:,400, 400, :])
            ch_i = ch_i[0] + 1j * ch_i[1]
            # print(ch_i[400,400,:])
            # ch_i = ch_i[:, :, ::model_factor]
        ch[:, :, :, i] = ch_i
        # plt.matshow(chv_norm(ch[:, :, :, i]))
        # plt.show()
    A1 = np.linalg.norm(np.sum(ch, axis=-1), axis=-1)
    A2 = np.sum(np.linalg.norm(ch, axis=-2).squeeze(), axis=-1)
    # plt.matshow(A1)
    # plt.colorbar()
    # plt.show()
    # plt.matshow(A2)
    # plt.colorbar()
    # plt.show()

    epsilon = 0.01
    plt.matshow(A1 / (A2 + epsilon))
    plt.colorbar()
    plt.show()

    plt.matshow(beta.cdf(A1 / (A2 + epsilon), 5, 1))
    plt.colorbar()
    plt.show()

    return A1 / (A2 + epsilon)


if __name__ == "__main__":
    N = 7
    w = 4
    sigma = 0.65
    scales = 4
    scale_factor = 2

    im = skio.imread("test/camvid.png")
    # im = skt.rescale(im, [0.25, 0.25, 1])
    phase_congruency(im,
                     N=N,
                     w=w,
                     sigma=sigma,
                     scales=scales,
                     scale_factor=scale_factor)

    U = sinusoid_pair(N, width=0.1)
    phase_congruency(im,
                     N=N,
                     w=w,
                     sigma=sigma,
                     scales=scales,
                     scale_factor=scale_factor,
                     model=U,
                     model_factor=2)