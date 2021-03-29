import skimage.io as skio
import skimage.color as skcolor
import matplotlib.pyplot as plt
import colorcet as cc

import chvector.transforms.chv as chv
from chvector.filters import log_gabor_spectrum
from chvector.models.archetypes import *
from chvector.models.solver import *

s = sinusoid_pair(7)
print(s)

im = skcolor.rgb2gray(skio.imread("COB3-orig.png"))
skio.imshow(im)
plt.show()
plt.pause(0)

gf = log_gabor_spectrum(im.shape[:2], 4, 0.5)
ch = chv.img_chv(im, gf, 7)

poly = create_poly(ch, s)

mag = np.linalg.norm(ch, axis=2)
plt.imshow(np.log(mag), cmap=cc.cm.fire)
plt.show()
plt.pause(0)
