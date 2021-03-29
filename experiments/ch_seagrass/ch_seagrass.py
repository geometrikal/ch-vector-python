import skimage.io as skio
import skimage.color as skcolor
import matplotlib.pyplot as plt
import colorcet as cc

import chvector.transforms.chv as chv
from chvector.filters import log_gabor_spectrum
from chvector.models.solver import *

filename = "D:\\Datasets\\Seagrass\\SeagrassFramesAll\\Strappy\\8FEB20-DSC02671.JPG"
im = skcolor.rgb2gray(skio.imread(filename))
skio.imshow(im)
plt.show()
plt.pause(0)

gf = log_gabor_spectrum(im.shape[:2], 16, 0.1)
gf = np.ones(gf.shape)
gf[0, 0] = 0
ch = chv.img_chv(im, gf, 1)

mag = np.linalg.norm(ch, axis=2)
plt.imshow(np.log(mag), cmap=cc.cm.fire)
plt.show()
plt.pause(0)

plt.imshow(np.abs(ch[:,:,0]), cmap=cc.cm.fire)
plt.show()
plt.pause(0)

plt.imshow(np.abs(ch[:,:,1]), cmap=cc.cm.fire)
plt.show()
plt.pause(0)