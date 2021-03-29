import skimage.io as skio
import skimage.color as skcolor
import matplotlib.pyplot as plt
import colorcet as cc

import chvector.transforms.chv as chv
from chvector.filters import log_gabor_spectrum
from chvector.models.solver import *

filename = r"D:\Development\Microfossil\ch-vector-python\experiments\cots\images\20200716_152526713_frame_0002597.jpg"
filename = r"D:\Development\Microfossil\ch-vector-python\experiments\cots\images\20200716_144842786_frame_0003338.jpg"
im = skcolor.rgb2gray(skio.imread(filename))
skio.imshow(im)
plt.axis("off")
plt.show()

N = 3

gf = log_gabor_spectrum(im.shape[:2], 32, 0.5)
ch = chv.img_chv(im, gf, N)

mag = np.linalg.norm(ch, axis=2)
plt.imshow(np.log(mag), cmap=cc.cm.fire)
plt.axis("off")
plt.show()
plt.pause(0)


mag0 = np.abs(ch[:,:,N])

plt.imshow(mag0, cmap=cc.cm.fire)
plt.axis("off")
plt.show()

plt.imshow(mag0 / mag, cmap=cc.cm.fire)
plt.axis("off")
plt.show()

# for i in range(N+1):
#     plt.imshow(np.abs(ch[:,:,i]), cmap=cc.cm.fire)
#     plt.show()
#     plt.pause(0)