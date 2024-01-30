import skimage.morphology as skmph
import skimage.io as skio
import skimage.color as skcolor
import skimage.util as skutil
import matplotlib.pyplot as plt
import chvector.utils.plotting as pl

import os

import scipy.ndimage as ndi

import chvector.transforms.chv as chv
from chvector.filters import log_gabor_spectrum
from chvector.models.archetypes import *
from chvector.models.solver import *
import chvector.models.weights as weights

import skimage as sk
from skimage.filters.rank import median

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

def noise_fill(imo, threshold, sigma):

    # OOB aware filter
    im_zeros = 0 * imo
    im_zeros[imo < threshold] = 1
    im_mask = imo.copy()
    im_mask[imo >= threshold] = 0
    im_filled = sk.filters.gaussian(im_mask, 31) / sk.filters.gaussian(im_zeros, 31)
    im_filled[im_filled > 1] = 1

    # Fill noise
    mask = imo < threshold
    mask = ndi.binary_fill_holes(mask)
    mask = skmph.binary_erosion(mask, skmph.disk(sigma*2))
    mn = np.mean(imo[mask])
    st = np.std(imo[mask])
    mask = ndi.gaussian_filter(mask.astype(float), sigma)
    pl.plot_image(mask)

    ns = skutil.noise.random_noise(np.zeros(imo.shape), mean=0, var=st ** 2)
    im = (ns + im_filled) * (1 - mask) + imo * mask
    return im


if __name__ == '__main__':

    os.makedirs('output', exist_ok=True)


    # Load image
    imo = skcolor.rgb2gray(skio.imread("cafewall.png"))
    imo = skcolor.rgb2gray(skio.imread("COB3-orig.png"))
    pl.plot_image(imo, save="output/orig.png")

    im = noise_fill(imo, 0.9, 5)
    pl.plot_image(im, save="output/noise.png")

    # Calculate CH vector
    # Number of RT orders
    N = 7
    # Filter wavelengths
    wavelengths = [4, 8, 16, 32]
    scales = len(wavelengths)
    ch_shape = (scales,) + im.shape + (2 * N + 1,)
    model_shape = (scales,) + im.shape
    # Filter bandwidth (sigma)
    sigma = 0.65
    # CH vector weights
    weightV = weights.sinusoid_weights(N, 0, 0)
    print("Weights are: ")
    print(weightV.transpose())
    # CH vector output array
    ch = np.zeros(ch_shape, dtype=complex)
    # Calculate
    for i, w in enumerate(wavelengths):
        print("Calculating CH vector for wavelength {}".format(w))
        if i == 0:
            passband = 'high'
        else:
            passband = 'band'
        gf = log_gabor_spectrum(im.shape, w, sigma)
        ch[i] = chv.img_chv(im, gf, N, weightV)

    # Calculate model parameters
    # Model vectors
    U = sinusoid_pair(N, weightV)

    # R = np.exp(1j*np.arange(-N,N+1)*np.pi/2)
    # U2 = R * U
    # U = np.concatenate((U,U2))
    # Test
    # cht = U[1]
    # cht = cht[np.newaxis, np.newaxis, :]
    # poly, delt, lamb = create_poly(cht, U)
    # poly = poly[:, :, ::2]
    # roots = poly_roots(poly)
    # th = np.angle(roots)
    # val_at_roots = poly_value(poly, th)
    # max_idx = np.argmax(val_at_roots, axis=2)
    # th_max = np.take_along_axis(th, max_idx[:, :, np.newaxis], axis=2)
    # # th_max = np.mod(th_max/2, np.pi)
    # th_max = th_max/2
    # s0 = poly_value(lamb[0], th_max)
    # s1 = poly_value(lamb[1], th_max)
    #
    # ch_model_t = poly_make(U[0], s0, th_max) + poly_make(U[1], s1, th_max)
    # # Residual
    # residual_t = cht - ch_model_t
    #
    # print(s0)
    # print(s1)




    # Model output array
    A_model = np.zeros(model_shape)
    phi_model = np.zeros(model_shape)
    theta_model = np.zeros(model_shape)
    A_model_2 = np.zeros(model_shape)
    phi_model_2 = np.zeros(model_shape)
    theta_model_2 = np.zeros(model_shape)
    # CH vector of model
    ch_model = np.zeros(ch_shape, dtype=complex)
    ch_residual = np.zeros(ch_shape, dtype=complex)
    ch_model_2 = np.zeros(ch_shape, dtype=complex)
    ch_residual_2 = np.zeros(ch_shape, dtype=complex)
    # Calculate
    for i, ch_channel in enumerate(ch):
        s, th, ch_model[i], ch_residual[i], poly, delt, lamb = solve_model(ch_channel, U, 2)
        A_model[i], phi_model[i], theta_model[i] = sinusoid_parameters(s[0], s[1], th)

        s, th, ch_model_2[i], ch_residual_2[i], _, _, _ = solve_model(ch_residual[i], U, 2)
        A_model_2[i], phi_model_2[i], theta_model_2[i] = sinusoid_parameters(s[0], s[1], th)

        g = poly_graph(poly[100,100,:])
        plt.plot(np.real(g))
        plt.show()

        # # Create response polynomials
        # poly, delt, lamb = create_poly(ch[:, :, :, i], U)
        # # Solve for maximum
        # # Sinusoid has poly order of 2
        # poly = poly[:, :, ::2]
        # roots = poly_roots_parallel(poly)
        # th = np.angle(roots)
        # val_at_roots = poly_value(poly, th)
        # max_idx = np.argmax(val_at_roots, axis=2)
        # th_max = np.take_along_axis(th, max_idx[:, :, np.newaxis], axis=2)
        # th_max = np.mod(th_max / 2, np.pi)
        # th_max /= 2
        # # Model component amplitudes
        # s0 = poly_value(lamb[0], th_max)
        # s1 = poly_value(lamb[1], th_max)
        # # Model parameters
        # A_model[:, :, i] = np.sqrt(s0 ** 2 + s1 ** 2)[:, :, 0] * np.sqrt(2)
        # phi_model[:, :, i] = np.arctan2(s1, s0)[:, :, 0]
        # theta_model[:, :, i] = th_max[:, :, 0]
        # # Model components
        # ch_model[:, :, :, i] = poly_make(U[0], s0, th_max) + poly_make(U[1], s1, th_max)
        # # Residual
        # residual[:, :, :, i] = ch[:, :, :, i] - ch_model[:, :, :, i]
        #
        # #pl.plot_image(s0[:, :, 0])
        # #pl.plot_image(s1[:, :, 0])
        # pl.plot_image(A_model[:, :, i])
        # pl.plot_phase(phi_model[:, :, i])
        # pl.plot_angle(theta_model[:, :, i])
        #
        # Achv = np.linalg.norm(ch[i], axis=2)
        # Achm = np.linalg.norm(ch_model[i], axis=2)
        # Ares = np.linalg.norm(ch_residual[i], axis=2)
        # Achm2 = np.linalg.norm(ch_model_2[i], axis=2)
        # Ares2 = np.linalg.norm(ch_residual_2[i], axis=2)
        #
        # pl.plot_image(Achv)
        # pl.plot_image(Achm)
        # pl.plot_image(Ares)
        # pl.plot_image(Achm2)
        # pl.plot_image(Ares2)
        #
        # pl.plot_image(s[0])
        # pl.plot_image(s[1])
        pl.plot_image(A_model[i], save="output/A_{}_1.png".format(i))
        pl.plot_phase(phi_model[i], save="output/phi_{}_1.png".format(i))
        pl.plot_angle(theta_model[i], save="output/theta_{}_1.png".format(i))

        pl.plot_image(A_model_2[i], save="output/A_{}_2.png".format(i))
        pl.plot_phase(phi_model_2[i], save="output/phi_{}_2.png".format(i))
        pl.plot_angle(theta_model_2[i], save="output/theta_{}_2.png".format(i))

        pl.plot_amp_angle(A_model[i], theta_model[i], save="output/Atheta_{}_1.png".format(i))
        pl.plot_amp_angle(A_model_2[i], theta_model_2[i], save="output/Atheta_{}_2.png".format(i))


    import skimage.morphology as morph
    def median_2d(th, radius=2):
        se = morph.disk(radius)


        return ndi.generic_filter(th, np.median, footprint=se)

    th = median_2d(theta_model[0], 11)
    pl.plot_angle(th)
    th = median_2d(th, 11)
    pl.plot_angle(th)