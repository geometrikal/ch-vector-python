import numpy as np
import chvector.filters as filt
import chvector.transforms.ops as ops
import chvector.models.weights as chw
# from scipy.fftpack import ifftshift
import tensorflow as tf


def tf_img_chv(im, basis_filter_spectrum, N, weights='sinusoid', is_fft=None):
    # basis_filter_spectrum = np.ones(basis_filter_spectrum.shape)
    print("FT")
    # CH vector weights
    if isinstance(weights, str) and weights == 'sinusoid':
        weights = chw.sinusoid_weights(N)
    if weights is None:
        weights = np.ones(2*N+1) / (2*N+1)
    print("Weights: {}".format(weights))

    multichannel = False
    if np.ndim(im) == 3:
        imt = im.transpose([2,0,1])
        basis_filter_spectrum = np.repeat(basis_filter_spectrum[np.newaxis, ...], im.shape[-1], axis=0)
        multichannel = True
    else:
        imt = im

    # Get fft of image (unless passed fft directly!)
    if is_fft is not None:
        f = imt
    else:
        f = tf.signal.fft2d(imt)
    f = tf.multiply(f, basis_filter_spectrum)

    # Calculate each order
    rts = []
    for n in range(-N, N+1):
        spectrum = rt_spectrum(im.shape[:2], n) * weights[n + N]
        rts.append(spectrum)
    rts = np.asarray(rts)
    if multichannel:
        rts = np.repeat(rts[:, np.newaxis, ...], im.shape[-1], axis=1)
    f = tf.repeat([f], 2 * N + 1, axis=0)
    ts = f * rts
    if multichannel:
        ch = tf.signal.ifft2d(ts).numpy().transpose([2,3,1,0])
    else:
        ch = tf.signal.ifft2d(ts).numpy().transpose([1, 2, 0])
    return ch


# def rt_spectrum(shape, n):
#     if n == 0:
#         return np.ones(shape)
#
#     # Get spectrum coordinates
#     ux, uy, _, _ = filt.fft_mesh(shape)
#
#     # Calculate RT spectrum
#     if n < 0:
#         v = ux - 1j * uy
#         n = -n
#     else:
#         v = ux + 1j * uy
#     r = np.abs(v)
#     ops.dc_to_value(r, 1)
#     v = (v / r) ** n
#     ops.dc_to_zero(v)
#     return v


def img_chv(im, basis_filter_spectrum, N, weights='sinusoid', is_fft=None):
    # basis_filter_spectrum = np.ones(basis_filter_spectrum.shape)
    # CH vector weights
    if isinstance(weights, str) and weights == 'sinusoid':
        weights = chw.sinusoid_weights(N)
    if weights is None:
        weights = np.ones(2*N+1) / (2*N+1)

    # If multi-channel, run on each channel
    if np.ndim(im) > 2:
        if im.shape[2] > 1:
            r = np.zeros(im.shape + (2*N+1,), dtype=np.complex)
            for i in range(im.shape[2]):
                r[..., i, :] = img_chv(im[..., i], basis_filter_spectrum, N, weights)
            return r
        else:
            # Remove last channel dim
            im = im[:, :, 0]

    # Get fft of image (unless passed fft directly!)
    if is_fft:
        f = im
    else:
        f = filt.img_fft2(im)
    f *= basis_filter_spectrum

    # CHV placeholder to store results
    ch = np.zeros(f.shape + (2*N+1,), dtype=np.complex)

    # Calculate each order
    for n in range(-N, N+1):
        t = rt_spectrum(f.shape, n) * weights[n + N]
        ch[:, :, n + N] = filt.img_ifft2(f * t)
    return ch


def rt_spectrum(shape, n):
    if n == 0:
        return np.ones(shape)

    # Get spectrum coordinates
    ux, uy, r, th = filt.fft_mesh(shape)

    # Calculate RT spectrum
    if n < 0:
        v = ux - 1j * uy
        n = -n
    else:
        v = ux + 1j * uy
    ops.dc_to_value(r, 1)
    v = np.power(v / r, n)
    ops.dc_to_zero(v)
    # Fix to set wavelngth 2 parts to zero so that conjugate looks correct
    if v.shape[0] % 2 == 0:
        v[v.shape[0] // 2, :] = 0
    if v.shape[1] % 2 == 0:
        v[:, v.shape[1] // 2] = 0
    return v


def chv_norm(chv):
    return np.linalg.norm(chv, axis=-1)


if __name__ == "__main__":
    import time
    from chvector.filters import log_gabor_spectrum

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    lg = log_gabor_spectrum((512, 512), 32, 0.5)

    for i in range(10):
        test_im = np.random.rand(512, 512, 3)

        print("CPU:")
        st = time.time()
        ch_vector = img_chv(test_im, lg, 7)
        print("{} seconds".format(time.time() - st))
        test_im = np.random.rand(512, 512, 3)

        print("GPU:")
        st = time.time()
        tf_ch_vector = tf_img_chv(test_im, lg, 7)
        print("{} seconds".format(time.time() - st))
        print(np.sum(ch_vector - tf_ch_vector))
