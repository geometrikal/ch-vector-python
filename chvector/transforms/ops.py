import numpy as np


def dc_to_value(spectrum, value):
    if np.ndim(spectrum) == 2:
        spectrum[0, 0] = value
    elif np.ndim(spectrum) == 3:
        spectrum[0, 0, :] = value


def dc_to_zero(spectrum):
    dc_to_value(spectrum, 0)


