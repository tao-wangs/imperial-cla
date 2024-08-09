import numpy as np
import timeit
import numpy.random as random

# pre-construct a matrix in the namespace to use in tests
random.seed(1651)
A0 = random.randn(500, 500)
x0 = random.randn(500)

def basic_matvec(A, x):
    """
    Elementary matrix-vector multiplication.

    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    returns an m-dimensional numpy array which is the product of A with x

    This should be implemented using a double loop over the entries of A

    :return b: m-dimensional numpy array
    """

    m, n = A.shape
    b = np.zeros(m)
    for i in range(m):
        for j in range(n):
            b[i] += A[i, j] * x[j]
    return b

def column_matvec(A, x):
    """
    Matrix-vector multiplication using the representation of the product
    Ax as linear combinations of the columns of A, using the entries in 
    x as coefficients.


    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    :return b: an m-dimensional numpy array which is the product of A with x

    This should be implemented using a single loop over the entries of x
    """

    # iterate through the columns of a (there are n columns)
    # multiply each column of a by the coefficient being the ith entry of x and add it to the result. 
    # b is a linear combination of the columns a, with coefficients being the entries of x. 
    m, n = A.shape
    b = np.zeros(m)
    for i in range(n):
        b += A[:, i] * x[i]
    return b 

def timeable_basic_matvec():
    """
    Doing a matvec example with the basic_matvec that we can
    pass to timeit.
    """

    b = basic_matvec(A0, x0) # noqa


def timeable_column_matvec():
    """
    Doing a matvec example with the column_matvec that we can
    pass to timeit.
    """

    b = column_matvec(A0, x0) # noqa


def timeable_numpy_matvec():
    """
    Doing a matvec example with the builtin numpy matvec so that
    we can pass to timeit.
    """

    b = A0.dot(x0) # noqa


def time_matvecs():
    """
    Get some timings for matvecs.
    """

    print("Timing for basic_matvec")
    print(timeit.Timer(timeable_basic_matvec).timeit(number=1))
    print("Timing for column_matvec")
    print(timeit.Timer(timeable_column_matvec).timeit(number=1))
    print("Timing for numpy matvec")
    print(timeit.Timer(timeable_numpy_matvec).timeit(number=1))


def rank2(u1, u2, v1, v2):
    """
    Return the rank2 matrix A = u1*v1^* + u2*v2^*.

    :param u1: m-dimensional numpy array
    :param u2: m-dimensional numpy array
    :param v1: n-dimensional numpy array
    :param v2: n-dimensional numpy array
    """

    #Note to self: 
    # np.outer() only does the transpose so if you want the hermition you must conjugate first. 

    B = np.outer(u1, v1.conj()) 
    C = np.outer(u2, v2.conj())

    return B + C 


def rank1pert_inv(u, v):
    """
    Return the inverse of the matrix A = I + uv^*, where I
    is the mxm dimensional identity matrix, with

    :param u: m-dimensional numpy array
    :param v: m-dimensional numpy array
    """

    m = u.shape[0]
    I = np.identity(m)    

    #Note to self:
    # A^-1 = I - alpha * uv^*, where alpha is -1/(1+v^*u)
    alpha = -1 / (1 + np.vdot(v, u))
    Ainv = I + alpha * np.outer(u, v.conj())

    return Ainv


def timeable_rank1pert_inv():
    """
    Doing an invmat example with the rank1pert_inv so that
    we can pass to timeit.
    """
    u = random.randn(400)
    v = random.randn(400)

    Ainv = rank1pert_inv(u, v)


def timeable_numpy_inv():
    """
    Doing an invmat example with the rank1pert_inv so that
    we can pass to timeit.
    """

    u = random.randn(400)
    v = random.randn(400)
    A = I = np.outer(u, v.conj())
    Ainv = np.linalg.inv(A)


def time_invmats():
    """
    Get some timings for rank1pert_inv
    """

    print("Timing for rank1pert_inv")
    print(timeit.Timer(timeable_rank1pert_inv).timeit(number=1))
    print("Timing for numpy inv")
    print(timeit.Timer(timeable_numpy_inv).timeit(number=1))


def ABiC(Ahat, xr, xi):
    """Return the real and imaginary parts of z = A*x, where A = B + iC
    with

    :param Ahat: an mxm-dimensional numpy array with Ahat[i,j] = B[i,j] \
    for i>=j and Ahat[i,j] = C[i,j] for i<j.

    :return zr: m-dimensional numpy arrays containing the real part of z.
    :return zi: m-dimensional numpy arrays containing the imaginary part of z.
    """

    #A = B + iC so Ax = (B+iC)x = Bx + iCx

    m, _ = Ahat.shape
    zr = np.zeros(m)
    zi = np.zeros(m)

    for i in range(m):
        #Grab column with correct entries from i to m, while i>=j
        real_col = Ahat[i:,i]
        #Grab missing B entries from A, they're now in their transpose location e.g. B[i][j] is in B[j][i] if i<j 
        real_missing = Ahat[i, 0:i]
        #Join them together to form real part of A
        ar = np.append(real_missing, real_col)
        
        #Grab row with correct entires from i to m 
        imag_col = -Ahat[i, i+1:]
        #Grab missing C entries from A, they're now in their transpose location e.g C[i][j] is in -A[j][i] if i>j
        #Keep in mind that diagonal (i=j) entires of a hermitian matrix are real, so imaginary component is 0
        imag_missing = np.append(Ahat[0:i, i],[0])

        #Reconstruct imaginary part 
        ai = np.append(imag_missing, imag_col)

        zr += ar*xr[i] - ai*xi[i] #real*real + imag*imag
        zi += ar*xi[i] + ai*xr[i] #real*imag + imag*real 

    return zr, zi
    #Ideally you want to update the actual real and imaginary parts of each column. 