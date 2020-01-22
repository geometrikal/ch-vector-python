import numpy as np
import scipy.ndimage as nd
from numba import jit


def create_poly(chv, U):
    """
    Create the polynomial we must find the max of
    :param chv: Image circular harmonic vector (along 3rd axis)
    :param U: Model circular harmonic vector set (vector along 2nd axes)
    :return: Response polynomial in theta
    """
    Uplus = np.linalg.pinv(np.transpose(U))
    poly = np.zeros((U.shape[0], chv.shape[0], chv.shape[1], chv.shape[2]*2-1), dtype=np.complex)
    delt = np.zeros((U.shape[0], chv.shape[0], chv.shape[1], chv.shape[2]), dtype=np.complex)
    lamb = np.zeros((U.shape[0], chv.shape[0], chv.shape[1], chv.shape[2]), dtype=np.complex)

    # Iterate over model parts
    for m in range(U.shape[0]):
        dv = np.conj(U[m])
        lv = Uplus[m]
        dv = dv[np.newaxis, np.newaxis, :]
        lv = lv[np.newaxis, np.newaxis, :]

        delt[m] = chv * dv
        lamb[m] = chv * lv
        poly[m] = conv_3rd_dim(delt[m], lamb[m])

    poly = np.sum(poly, axis=0)
    return poly, delt, lamb


def conv_3rd_dim(A, B):
    res = np.zeros((A.shape[0], A.shape[1], A.shape[2]*2-1), dtype=np.complex)
    for i in range(A.shape[0]):
        print(i)
        for j in range(A.shape[1]):
            v1 = A[i,j,:].flatten()
            v2 = B[i,j,:].flatten()
            v3 = np.convolve(v1,v2)
            res[i, j, :] = v3
    return res


def poly_roots(A):
    roots = np.zeros((A.shape[0], A.shape[1], A.shape[2]-1), dtype=np.complex)
    k = np.arange(-(A.shape[2] - A.shape[2] // 2 - 1), A.shape[2] // 2 + 1)
    k = k[np.newaxis, np.newaxis, :]
    A = A * k
    for i in range(A.shape[0]):
        print(i)
        for j in range(A.shape[1]):
            p = A[i, j, :].flatten()
            roots[i, j, :] = np.roots(p)
    return roots


def poly_value(p, theta, isreal=True):
    v = np.zeros((p.shape[0], p.shape[1], theta.shape[2]), dtype=np.complex)

    for i in range(theta.shape[2]):
        k = np.arange(-(p.shape[2] - p.shape[2] // 2 - 1), p.shape[2] // 2 + 1)
        k = k[np.newaxis, np.newaxis, :]
        k = np.tile(k, (p.shape[0], p.shape[1], 1))
        k = np.exp(1j * k * -theta[:,:,i:i+1])
        v[:,:,i] = np.sum(p * k, axis=2)

    if isreal:
        v = np.real(v)
    return v





def polymatmult(A, B):
    print("np.shape(A)", np.shape(A))
    print("np.shape(B)", np.shape(B))
    [NAx, NAy, NAz] = np.shape(A)
    [NBx, NBy, NBz] = np.shape(B)

    Deg = NAz + NBz - 1
    print("Deg", Deg)
    C = np.zeros((Deg, NAx, NBy))

    m, n = np.triu_indices(NBz, 0, Deg)
    m, n = m[n - m < NAz], n[n - m < NAz]
    np.add.at(C, n,  np.moveaxis(A[:, :, (n - m)], -1, 0) @ np.moveaxis(B[:, :, m], -1, 0))

    return np.moveaxis(C, 0, -1)

def polmatmultslow(A, B):
    """polmatmult(A,B)
    multiplies two polynomial matrices (arrays) A and B, where each matrix entry is a polynomial.
    Those polynomial entries are in the 3rd dimension
    The third dimension can also be interpreted as containing the (2D) coefficient matrices of exponent of z^-1.
    Result is C=A*B;"""
    print("np.shape(A)", np.shape(A))
    print("np.shape(B)", np.shape(B))
    [NAx, NAy, NAz] = np.shape(A);
    [NBx, NBy, NBz] = np.shape(B);

    "Degree +1 of resulting polynomial, with NAz-1 and NBz-1 being the degree of the input  polynomials:"

    Deg = NAz + NBz - 1;
    print("Deg", Deg)
    C = np.zeros((NAx, NBy, Deg));

    "Convolution of matrices:"
    for n in range(0, (Deg)):
        for m in range(0, n + 1):
            if ((n - m) < NAz and m < NBz):
                C[:, :, n] = C[:, :, n] + np.dot(A[:, :, (n - m)], B[:, :, m]);

    return C