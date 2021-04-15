from chvector.filters import log_gabor_spectrum
from chvector.transforms.chv import img_chv, chv_norm
import numpy as np
import skimage.color as skc


def scale(im, N, w, sigma, scales, scale_factor=2):
    if np.ndim(im) > 2:
        im = skc.rgb2gray(im)
    ch = np.zeros(im.shape + (2 * N + 1, scales), dtype=np.complex)

    A = []
    for i in range(scales):
        spectrum = log_gabor_spectrum(im.shape[:2], w * scale_factor ** i, sigma, passband='band')
        ch_i = img_chv(im, spectrum, N)
        A.append(chv_norm(ch_i))
    A = np.asarray(A)

    return np.max(A, axis=0), np.argmax(A, axis=0)


if __name__ == "__main__":
    import skimage.io as skio
    import matplotlib.pyplot as plt

    N = 7
    w = 4
    sigma = 0.65
    scales = 4
    scale_factor = 2



    im = skio.imread("test/camvid.png")
    val, idx = scale(im,
                     N=N,
                     w=w,
                     sigma=sigma,
                     scales=scales,
                     scale_factor=scale_factor)

    plt.matshow(im)
    plt.show()

    plt.matshow(val)
    plt.show()

    plt.matshow(idx)
    plt.show()
