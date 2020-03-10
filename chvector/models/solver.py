import numpy as np
import chvector.utils.parallel as pll
import time
import scipy.ndimage as nd


# from numba import jit


def create_poly(chv, U):
    """
    Create the polynomial we must find the max of
    :param chv: Image circular harmonic vector (along 3rd axis)
    :param U: Model circular harmonic vector set (vector along 2nd axes)
    :return: Response polynomial in theta
    """
    Uplus = np.linalg.pinv(U.transpose())
    poly = np.zeros((U.shape[0], chv.shape[0], chv.shape[1], chv.shape[2] * 2 - 1), dtype=np.complex)
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


def conv_3rd_dim(A, B, parallel=True):
    if parallel is False:
        res = np.zeros((A.shape[0], A.shape[1], A.shape[2] * 2 - 1), dtype=np.complex)
        print("Convolve along 3rd dimension...")
        for i in range(A.shape[0]):
            print("\r - row {} / {}".format(i + 1, A.shape[0]), end="")
            for j in range(A.shape[1]):
                v1 = A[i, j, :].flatten()
                v2 = B[i, j, :].flatten()
                v3 = np.convolve(v1, v2)
                res[i, j, :] = v3
        print()
        return res
    else:
        print("Convolving along 3rd dimension (parallel)... ", end="")
        start = time.time()
        M = np.stack((A, B), 3)
        res = pll.parallel_chunked(func_conv_3rd_dim_parallel, M)
        stop = time.time()
        print("({}s)".format(stop - start))
        return res


def func_conv_3rd_dim_parallel(M):
    A = M[:, :, :, 0]
    B = M[:, :, :, 1]
    res = np.zeros((A.shape[0], A.shape[1], A.shape[2] * 2 - 1), dtype=np.complex)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v1 = A[i, j, :].flatten()
            v2 = B[i, j, :].flatten()
            v3 = np.convolve(v1, v2)
            res[i, j, :] = v3
    return res


def poly_roots(A):
    print("Solving polynomial roots... ", end="")
    start = time.time()
    k = np.arange(-(A.shape[2] - A.shape[2] // 2 - 1), A.shape[2] // 2 + 1)
    k = k[np.newaxis, np.newaxis, :]
    A = A * k
    roots = np.apply_along_axis(np.roots, 2, A)
    stop = time.time()
    print("({}s)".format(stop - start))
    return roots


def poly_roots_parallel(A):
    print("Solving polynomial roots (parallel)...", end="")
    start = time.time()
    k = np.arange(-(A.shape[2] - A.shape[2] // 2 - 1), A.shape[2] // 2 + 1)
    k = k[np.newaxis, np.newaxis, :]
    A = A * k
    roots = pll.parallel_apply_along_axis(np.roots, 2, A)
    stop = time.time()
    print("({}s)".format(stop - start))
    return roots


def poly_value(p, theta, isreal=True):
    v = np.zeros((p.shape[0], p.shape[1], theta.shape[2]), dtype=np.complex)
    N = p.shape[2] // 2
    for i in range(theta.shape[2]):
        k = np.arange(-N, N + 1)
        k = k[np.newaxis, np.newaxis, :]
        k = np.tile(k, (p.shape[0], p.shape[1], 1))
        k = np.exp(1j * k * -theta[:, :, i:i + 1])
        v[:, :, i] = np.sum(p * k, axis=2)
    if isreal:
        v = np.real(v)
    return v


def poly_graph(p, th_range=(0, 2*np.pi), num_points=360):
    p = p.flatten()
    N = len(p) // 2
    p = np.tile(p, (num_points, 1))
    k = np.arange(-N, N + 1)
    k = np.tile(k, (num_points, 1))
    th = np.linspace(th_range[0],th_range[1], num_points)[:,np.newaxis]
    k = np.exp(1j * k * -th)
    return np.sum(p * k, axis=1).flatten()


def poly_make(vector, A, theta):
    shape = A.shape
    # Model vector into 3rd dimension
    v = vector.flatten()[np.newaxis, np.newaxis, :]
    M = np.tile(v, (shape[0], shape[1], 1))
    # Orders
    N = v.shape[2] // 2
    # Rotation matrix (in third dimension)
    k = np.arange(-N, N + 1, dtype=np.float)[np.newaxis, np.newaxis, :]
    k = np.tile(k, (shape[0], shape[1], 1))
    k *= theta
    R = np.exp(1j * k)
    # Apply rotation
    M *= R
    # Apply amplitude
    M *= A[:, :, np.newaxis]
    return M


def solve_model(ch, U, order):
    poly, delt, lamb = create_poly(ch, U)
    # Solve for maximum
    poly = poly[:, :, ::order]
    roots = poly_roots_parallel(poly)
    th = np.angle(roots)
    val_at_roots = poly_value(poly, th)
    max_idx = np.argmax(val_at_roots, axis=2)
    th_max = np.take_along_axis(th, max_idx[:, :, np.newaxis], axis=2)
    # th_max = np.mod(th_max / 2, np.pi)
    th_max /= order
    # Model component amplitudes
    s = np.zeros(lamb.shape[0:3])
    for i, m in enumerate(lamb):
        s[i] = poly_value(lamb[i], th_max)[:, :, 0]
    # Model components
    ch_model = np.zeros(ch.shape, dtype=np.complex)
    for i, m in enumerate(s):
        ch_model += poly_make(U[i], m, th_max)
    # Residual
    ch_residual = ch - ch_model
    # Return
    return s, th_max[:, :, 0], ch_model, ch_residual, poly, delt, lamb


def sinusoid_parameters(se, so, theta):
    # Model parameters
    A = np.sqrt(se ** 2 + so ** 2) * np.sqrt(2)
    phi = np.arctan2(so, se)
    return A, phi, theta


# def polymatmult(A, B):
#     print("np.shape(A)", np.shape(A))
#     print("np.shape(B)", np.shape(B))
#     [NAx, NAy, NAz] = np.shape(A)
#     [NBx, NBy, NBz] = np.shape(B)
#
#     Deg = NAz + NBz - 1
#     print("Deg", Deg)
#     C = np.zeros((Deg, NAx, NBy))
#
#     m, n = np.triu_indices(NBz, 0, Deg)
#     m, n = m[n - m < NAz], n[n - m < NAz]
#     np.add.at(C, n, np.moveaxis(A[:, :, (n - m)], -1, 0) @ np.moveaxis(B[:, :, m], -1, 0))
#
#     return np.moveaxis(C, 0, -1)
#
#
# def polmatmultslow(A, B):
#     """polmatmult(A,B)
#     multiplies two polynomial matrices (arrays) A and B, where each matrix entry is a polynomial.
#     Those polynomial entries are in the 3rd dimension
#     The third dimension can also be interpreted as containing the (2D) coefficient matrices of exponent of z^-1.
#     Result is C=A*B;"""
#     print("np.shape(A)", np.shape(A))
#     print("np.shape(B)", np.shape(B))
#     [NAx, NAy, NAz] = np.shape(A);
#     [NBx, NBy, NBz] = np.shape(B);
#
#     "Degree +1 of resulting polynomial, with NAz-1 and NBz-1 being the degree of the input  polynomials:"
#
#     Deg = NAz + NBz - 1;
#     print("Deg", Deg)
#     C = np.zeros((NAx, NBy, Deg));
#
#     "Convolution of matrices:"
#     for n in range(0, (Deg)):
#         for m in range(0, n + 1):
#             if ((n - m) < NAz and m < NBz):
#                 C[:, :, n] = C[:, :, n] + np.dot(A[:, :, (n - m)], B[:, :, m]);
#
#     return C
