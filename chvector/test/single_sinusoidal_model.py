import skimage.io as skio
import skimage.color as skcolor
import chvector.utils.plotting as pl

import chvector.transforms.chv as chv
from chvector.filters import log_gabor_spectrum
from chvector.models.archetypes import *
from chvector.models.solver import *
import chvector.models.weights as weights

'''
Example sinusoidal image model analysis of coral image

1. Calculate the CH vector for each frequency channel
2. Create the matrix U with the archetypes
3. Apply the weights to each
4. Solve for the model
'''
#
# im = skcolor.rgb2grey(skio.imread("monocircle512.tiff"))
# pl.plot_image(im)
#
# # Calculate CH vector
# # Number of RT orders
# N = 7
# # Filter bandwidth (sigma)
# sigma = 0.65
#
# # Create circular harmonic vector
# gf = log_gabor_spectrum(im.shape, 16, sigma)
# ch1 = chv.img_chv(im, gf, N, weights=weights.sinusoid(N,0,0))
#
# print(weights.sinusoid(N,0,0))
#
# # Create model wavelets (needs proper weighting)
# U = sinusoid_pair(N, weights=weights.sinusoid(N,0,0))
#
# # Create response polynomials
# poly, lamb, delt = create_poly(ch1, U)
#
# # Solve for maximum
# # Sinusoid has poly order of 2
# poly = poly[:,:,::2]
# roots = poly_roots(poly)
# th = np.angle(roots)
# val_at_roots = poly_value(poly, th)
# max_idx = np.argmax(val_at_roots, axis=2)
# th_max = np.take_along_axis(th, max_idx[:,:,np.newaxis], axis=2)
# th_max /= 2
#
# # Model component amplitudes
# s0 = poly_value(lamb[0], th_max)
# s1 = poly_value(lamb[1], th_max)
#
# Ar = np.sqrt(s0**2 + s1**2)[:,:,0]
# phi2 = np.arctan2(s1, s0)[:,:,0]
# th2 = th_max[:,:,0]
#
# skio.imshow(s0[:,:,0])
# plt.show()
# plt.pause(0)
#
# skio.imshow(s1[:,:,0])
# plt.show()
# plt.pause(0)
#
# skio.imshow(Ar)
# plt.show()
#
# fig = plt.figure(figsize=(40, 10))
# plt.imshow(phi2, cmap=cc.cm.cyclic_mrybm_35_75_c68_s25)
# plt.show()
#
# plt.imshow(th2, cmap=cc.cm.cyclic_mrybm_35_75_c68_s25)
# plt.show()
#
# th2_mean = median(sk.util.img_as_ubyte(th2 / (np.pi/2)), disk(20))
# plt.imshow(th2_mean, cmap=cc.cm.cyclic_mrybm_35_75_c68_s25)
# plt.show()



if __name__ == '__main__':

    # Load image
    im = skcolor.rgb2gray(skio.imread("COB3-orig.png"))
    # im = skcolor.rgb2grey(skio.imread("monocircle512.tiff"))
    pl.plot_image(im)

    # Calculate CH vector
    # Number of RT orders
    N = 7
    # Filter wavelengths
    wavelengths = [4, 8, 16, 32]
    # Filter bandwidth (sigma)
    sigma = 0.65
    # CH vector weights
    wv = weights.sinusoid_weights(N, 0, 0)
    print("Weights are: ")
    print(wv.transpose())
    # CH vector output array
    ch = np.zeros(im.shape + (2 * N + 1, len(wavelengths)), dtype=complex)
    # Calculate
    for i, w in enumerate(wavelengths):
        print("Calculating CH vector for wavelength {}".format(w))
        if i == 0:
            passband = 'high'
        else:
            passband = 'band'
        gf = log_gabor_spectrum(im.shape, w, sigma)
        ch[:, :, :, i] = chv.img_chv(im, gf, N, wv)

    # Calculate model parameters
    # Model vectors
    U = sinusoid_pair(N, w)
    # Model output array
    A_model = np.zeros(im.shape + (len(wavelengths), ))
    phi_model = np.zeros(im.shape + (len(wavelengths), ))
    theta_model = np.zeros(im.shape + (len(wavelengths), ))
    # Calculate
    for i in range(ch.shape[3]):
        # Create response polynomials
        poly, lamb, delt = create_poly(ch[:, :, :, i], U)
        # Solve for maximum
        # Sinusoid has poly order of 2
        poly = poly[:, :, ::2]
        roots = poly_roots(poly)
        th = np.angle(roots)
        val_at_roots = poly_value(poly, th)
        max_idx = np.argmax(val_at_roots, axis=2)
        th_max = np.take_along_axis(th, max_idx[:, :, np.newaxis], axis=2)
        th_max /= 2
        # Model component amplitudes
        s0 = poly_value(lamb[0], th_max)
        s1 = poly_value(lamb[1], th_max)
        # Model parameters
        A_model[:, :, i] = np.sqrt(s0 ** 2 + s1 ** 2)[:, :, 0]
        phi_model[:, :, i] = np.arctan2(s1, s0)[:, :, 0]
        theta_model[:, :, i] = th_max[:, :, 0]
        # Plots
        pl.plot_image(s0[:,:,0])
        pl.plot_image(s1[:,:,0])
        pl.plot_image(A_model[:, :, i])
        pl.plot_phase(phi_model[:, :, i])
        pl.plot_angle(theta_model[:, :, i])


    # Region growing