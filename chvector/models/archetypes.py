import numpy as np
import chvector.models.weights as chw

from chvector.models.weights import sinusoid_weights


def sinusoid(A, phi, theta, N, weights=None):
    if weights is None:
        weights = chw.sinusoid(N)
    vec = np.zeros([2*N+1], dtype=complex)
    for n in range(-N,N+1):
        idx = n+N
        if np.mod(n,2) == 0:
            vec[idx] = A * np.exp(1j * n * theta) * np.cos(phi)
        else:
            vec[idx] = A * np.exp(1j * n * theta) * 1j * np.sin(phi)
    if weights is None:
        vec = vec / np.linalg.norm(vec)
    else:
        vec = vec * weights
    return vec


def sinusoid_pair(N, weights=None, width=None):
    if weights is None and width is not None:
        weights = sinusoid_weights(N, width)
    mat = np.zeros([2, 2 * N + 1], dtype=complex)
    mat[0] = sinusoid(1, 0, 0, N, weights)
    mat[1] = sinusoid(1, np.pi/2, 0, N, weights)
    return mat



