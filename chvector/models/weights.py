import numpy as np


def sinusoid(N, width, mode):
    # %CHWEIGHTSSINUSOID - Weighting matrix for sinusoidal model that ensures odd
    # %                    and even components are weighted equally.
    # %
    # % Inputs:
    # %
    # % N         Maximum CH order to use
    # % width     Width in either radians (mode = 0) or normalised to N (mode = 1*)
    # %           For normalised (mode = 1) the following widths result in:
    # %           0 - each odd order (and each order) are weighted the same. the
    # %           angular respons has larger lobes
    # %           1 - the higher orders are weighted less, resulting in smaller
    # %           angular lobes
    # %           2 - the angular response is very smooth
    # %           (Note: the width is a continuous parameter, e.g. width = 1.3 is
    # %           perfectly fine)
    # % mode      0 - absolute width, 1* - normalised
    # %
    # % Start with width = 0 and adjust when necessary. A width of 0 gives the
    # % best noise response for a single sinusoidal model, but in multiple
    # % component models a higher width is sometimes better so that off-axis
    # % components have less influence.
    # %
    # % Outputs:
    # %weights
    # % weights   The weighting vector (1 x 2N+1)
    # %
    # %
    # % Written by:
    # %
    # % Ross Marchant
    # % James Cook University
    # % ross.marchant@my.jcu.edu.au
    # %

    # Use normalisation?
    if mode == 1:
        width = width * 5.64/N - 6.57/N^2

    # Width cannot be zero
    if width == 0:
        width = 0.00001

    [x,y] = np.meshgrid(np.arange(-N,N+1), np.arange(-N,N+1))
    dn = x - y

    # Solve for window
    V = 2*np.sin(width*dn/2) / dn
    V[np.isnan(V)] = 0
    V = V + 2*width * np.eye(2*N+1)
    V[np.mod(dn,2) == 1] = 0

    # Calculate eigenvectors
    [eigenvalues, eigenvectors] = np.linalg.eig(V)

    # Last two are the odd and even weighted sinusoidal CH vectors. Use their absolute values to get the weights.
    weights = np.sum(np.abs(eigenvectors[:, 0:2]) / np.sqrt(2), axis=1).transpose()

    return weights
