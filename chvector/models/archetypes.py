import numpy as np


def sinusoid(A, phi, theta, N, weights=None):
    vec = np.zeros([2*N+1], dtype=np.complex)
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


def sinusoid_pair(N, weights=None):
    mat = np.zeros([2, 2 * N + 1], dtype=np.complex)
    mat[0] = sinusoid(1, 0, 0, N, weights)
    mat[1] = sinusoid(1, np.pi/2, 0, N, weights)
    return mat



