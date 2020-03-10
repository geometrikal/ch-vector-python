import numpy as np
import skimage.morphology as morph
import skimage.filters as filt


def variance(im, radius=2):
    se = morph.disk(radius)
    return filt.generic_filter(im, np.var, footprint=se)