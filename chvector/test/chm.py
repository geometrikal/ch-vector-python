import matplotlib.pyplot as plt
import skimage.io as skio
from tqdm import tqdm
import numpy as np

from chvector.filters import scale_adaptive_spectrums
from chvector.models.solver import conv_3rd_dim, poly_roots, poly_value
from chvector.transforms.chv import img_chv



def complex_to_real(matrix, dim):
    matrix_0 = np.swapaxes(matrix, dim, 0)
    res = np.zeros(matrix_0.shape, dtype=np.complex)
    print(res.shape)
    N = (matrix_0.shape[0] - 1) // 2
    print(N)

    for i in range(N + 1):
        if i == 0:
            res[0] = np.real(matrix_0[N])
        else:
            res[i * 2 - 1] = np.real(matrix_0[N + i]) * np.sqrt(2)
            res[i * 2] = np.imag(matrix_0[N + i]) * np.sqrt(2)
    res = np.swapaxes(res, dim, 0)
    return res


def real_to_complex(matrix, dim, is_chv=True):
    matrix_0 = np.swapaxes(matrix, dim, 0)
    res = np.zeros(matrix_0.shape, dtype=np.complex)
    N = (matrix_0.shape[0] - 1) // 2
    for i in range(N + 1):
        if i == 0:
            res[N] = matrix_0[i]
        else:
            res[N + i] = (matrix_0[2 * i - 1] + 1j * matrix_0[2 * i]) / np.sqrt(2)
            if is_chv:
                res[N - i] = (-1) ** (-i) * (matrix_0[2 * i - 1] - 1j * matrix_0[2 * i]) / np.sqrt(2)
            else:
                res[N - i] = (matrix_0[2 * i - 1] - 1j * matrix_0[2 * i]) / np.sqrt(2)
    res = np.swapaxes(res, dim, 0)
    return res


if __name__ == "__main__":
    im = skio.imread("spiral.png", as_gray=True)  #[100:150, 100:150]

    shape = im.shape[:2]
    cutoff = 32
    M = 7
    N = 3

    h = scale_adaptive_spectrums(shape, cutoff, M)


    shm = np.zeros(shape + (2 * M + 1, 2 * N + 1), dtype=np.complex)

    for i in tqdm(range(h.shape[2])):
        # h_all = np.linalg.norm(h[:, :, i, :], axis=-1)
        # pair = np.hstack((h[:, :, i, 0], h[:, :, i, 1], h_all))
        # plt.matshow(pair)
        # plt.show()
        # plt.plot(pair[0, :shape[1]//2])
        # plt.plot(pair[0, shape[1]:shape[1] + shape[1]//2])
        # plt.show()

        chv = img_chv(im, h[:, :, i], N)
        shm[:, :, i, :] = chv

        # mag1 = conv_3rd_dim(shm[:, :, 2 * i, :], np.conj(shm[:, :, 2 * i, :]))
        # mag2 = conv_3rd_dim(shm[:, :, 2 * i + 1, :], np.conj(shm[:, :, 2 * i + 1, :]))

        # mag1 = np.linalg.norm(shm[:, :, 2 * i, :], axis=-1)
        # mag2 = np.linalg.norm(shm[:, :, 2 * i + 1, :], axis=-1)
        #
        # plt.imshow(mag1), plt.show()
        # plt.imshow(mag2), plt.show()
        #
        # mag = np.sqrt(mag1 ** 2 + mag2 ** 2)
        #
        # plt.imshow(mag), plt.show()

        # plt.imshow(np.real(shm[:, :, 2 * i, N+1])), plt.show()
        # plt.imshow(shm[:, :, 2 * i + 1, :]), plt.show()

    shmm = complex_to_real(shm,3)
    shmmm = real_to_complex(shmm,3)

    print(shm[0,0,0,:])
    print(shmm[0,0,0,:])
    print(shmmm[0,0,0,:])

    shmmv = real_to_complex(shmm, 2, False)
    cs = []
    for i in range(shmmv.shape[3]):
        v = shmmv[:,:,:,i]
        c = conv_3rd_dim(v, v, False)
        cs.append(c)

    cs = np.asarray(cs)
    css = cs.sum(axis=0)

    roots = poly_roots(css)
    th = np.angle(roots)
    val_at_roots = poly_value(css, th)
    max_idx = np.argmax(val_at_roots, axis=2)
    th_max = np.take_along_axis(th, max_idx[:, :, np.newaxis], axis=2)
    final_val = poly_value(css, th_max)

    plt.matshow(th_max), plt.colorbar(), plt.show()
