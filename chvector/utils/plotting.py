import matplotlib.pyplot as plt
import colorcet as cet
import numpy as np


def plot_image(image, title=None, vmin=None, vmax=None, cm=cet.cm.fire, colorbar=True, show=True, save=None):
    height = image.shape[0]
    width = image.shape[1]
    plt.figure(figsize=(width / 72 + 1, height / 72 + 1), dpi=72)
    plt.imshow(image, cmap=cm)
    if colorbar:
        colorbar_fixed(image.shape)
    if title:
        plt.title(title)
    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()


def plot_phase(image, title=None, vmin=-np.pi, vmax=np.pi, cm=cet.cm.cyclic_mrybm_35_75_c68_s25, colorbar=True, show=True, save=None):
    plot_image(image, title=title, vmin=vmin, vmax=vmax, cm=cm, colorbar=colorbar, show=show, save=save)


def plot_angle(image, title=None, vmin=-np.pi, vmax=np.pi, cm=cet.cm.cyclic_mrybm_35_75_c68_s25, colorbar=True, show=True, save=None):
    plot_image(image, title=title, vmin=vmin, vmax=vmax, cm=cm, show=show, save=save)


def plot_amp_angle(A, th, title=None, Amin=None, Amax=None, thmin=-np.pi, thmax=np.pi, cm=cet.cm.cyclic_mrybm_35_75_c68_s25, show=True, save=None):
    if Amin == None and Amax == None:
        Amin = np.min(A)
        Amax = np.max(A)
    elif Amin == None:
        Amin = np.min(A)
    elif Amax == None:
        Amax = np.max(A)

    A = (A - Amin) / (Amax - Amin)
    th = (th - thmin) / (thmax - thmin)
    c = cm(th)
    print(c[100, 100, :])
    c *= A[:, :, np.newaxis]
    c[:,:,3] = 1
    print(c[100,100,:])
    height = A.shape[0]
    width = A.shape[1]
    plt.figure(figsize=(width / 72 + 1, height / 72 + 1), dpi=72)
    plt.imshow(c)
    if title:
        plt.title(title)
    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()



def colorbar_fixed(image_shape):
    ratio = image_shape[0]/image_shape[1]
    plt.colorbar(fraction=0.046*ratio, pad=0.04)

