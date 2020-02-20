import numpy as np
import skimage.io as skio
import skimage.color as skcolor
import matplotlib.pyplot as plt
import colorcet as cc

import chvector.transforms.chv as chv
from chvector.filters.filters import log_gabor_spectrum
from chvector.models.archetypes import *
from chvector.models.solver import *
import chvector.models.weights as weights

import skimage as sk
from skimage.filters.rank import median
from skimage.morphology import disk

# Input
# Number of RT orders
N = 7
# Filter wavelength
wavelength = 8
# Filter bandwidth (sigma)
sigma = 0.5

# Load image
im = skcolor.rgb2gray(skio.imread("COB3-orig.png"))
#im = skcolor.rgb2gray(skio.imread("monocircle512.tiff"))
skio.imshow(im)
plt.show()

# Create circular harmonic vector
gf = log_gabor_spectrum(im.shape, wavelength, sigma)
ch = chv.img_chv(im, gf, N)

# Create model wavelets (needs proper weighting)
U = sinusoid_pair(N, weights=weights.sinusoid(N,0,0))

# Create response polynomials
poly, lamb, delt = create_poly(ch, U)

# Solve for maximum
# Sinusoid has poly order of 2
poly = poly[:,:,::2]
roots = poly_roots(poly)
th = np.angle(roots)
val_at_roots = poly_value(poly, th)
max_idx = np.argmax(val_at_roots, axis=2)
th_max = np.take_along_axis(th, max_idx[:,:,np.newaxis], axis=2)
th_max /= 2

# Model component amplitudes
s0 = poly_value(lamb[0], th_max)
s1 = poly_value(lamb[1], th_max)

Ar = np.sqrt(s0**2 + s1**2)[:,:,0]
phi2 = np.arctan2(s1, s0)[:,:,0]
th2 = th_max[:,:,0]

skio.imshow(s0[:,:,0])
plt.show()
plt.pause(0)

skio.imshow(s1[:,:,0])
plt.show()
plt.pause(0)

skio.imshow(Ar)
plt.show()

fig = plt.figure(figsize=(40, 10))
plt.imshow(phi2, cmap=cc.cm.cyclic_mrybm_35_75_c68_s25)
plt.show()

plt.imshow(th2, cmap=cc.cm.cyclic_mrybm_35_75_c68_s25)
plt.show()

th2_mean = median(sk.util.img_as_ubyte(th2 / (np.pi/2)), disk(20))
plt.imshow(th2_mean, cmap=cc.cm.cyclic_mrybm_35_75_c68_s25)
plt.show()