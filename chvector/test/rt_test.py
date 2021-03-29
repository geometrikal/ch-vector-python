from chvector.transforms.fft import *
from chvector.filters.rt import *

import matplotlib.pyplot as plt

ux,uy,r,th = fft_mesh(256,256)

t = nth_order_rt(ux,uy,3)

plt.imshow(fftshift(np.real(t))), plt.show()