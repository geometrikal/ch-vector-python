import matplotlib.pyplot as plt
import colorcet as cet
import numpy as np


def plot_image(image, title=None, vmin=None, vmax=None, cm=cet.cm.fire, colorbar=True, show=True):
    height = image.shape[0]
    width = image.shape[1]
    plt.figure(figsize=(width / 72 + 1, height / 72 + 1), dpi=72)
    plt.imshow(image, cmap=cm)
    if colorbar:
        plt.colorbar()
    if title:
        plt.title(title)
    if show:
        plt.show()


def plot_phase(image, title=None, vmin=-np.pi, vmax=np.pi, cm=cet.cm.cyclic_mrybm_35_75_c68_s25, colorbar=True, show=True):
    plot_image(image, title=title, vmin=vmin, vmax=vmax, cm=cm, show=show)


def plot_angle(image, title=None, vmin=-np.pi, vmax=np.pi, cm=cet.cm.cyclic_mrybm_35_75_c68_s25, colorbar=True, show=True):
    plot_image(image, title=title, vmin=vmin, vmax=vmax, cm=cm, show=show)

