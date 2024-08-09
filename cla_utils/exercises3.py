import numpy as np


def householder(A, kmax=None, complex=False):
    """
    Given a real mxn matrix A, find the reduction to upper triangular matrix R
    using Householder transformations. The reduction should be done "in-place",
    so that A is transformed to R.

    :param A: an mxn-dimensional numpy array
    :param kmax: an integer, the number of columns of A to reduce \
    to upper triangular. If not present, will default to n.
    :param complex: optional complex array handling
    """
        
    sign = lambda x: -1 if x < 0 else 1

    m, n = A.shape
    if kmax is None:
        kmax = n

    for k in range(kmax):
        x = A[k:, k].copy()
        if complex:
            x_norm = np.linalg.norm(x)
            e1 = np.eye(len(x), dtype=A.dtype)[0]
            angle = np.angle(x[0])
            v1 = x + np.exp(1j * angle) * x_norm * e1
            v2 = x - np.exp(1j * angle) * x_norm * e1 
            v = v1 if np.linalg.norm(v1) > np.linalg.norm(v2) else v2
        else: 
            x[0] += sign(x[0])*np.linalg.norm(x)
            v = x.copy()

        v /= np.linalg.norm(v)
        A[k:, k:] -= 2 * np.outer(v, np.dot(np.conj(v), A[k:, k:]))

        
def solve_U(U, b):
    """
    Solve systems Ux_i=b_i for x_i with U upper triangular, i=1,2,...,k

    :param U: an mxm-dimensional numpy array, assumed upper triangular
    :param b: an mxk-dimensional numpy array, with ith column containing 
       b_i
    :return x: an mxk-dimensional numpy array, with ith column containing 
       the solution x_i

    """    
    m, n = b.shape
    x = np.zeros([m, n], dtype=b.dtype)
    
    x[-1,:] = b[-1,:] / U[-1,-1]
    for i in range(m-2, -1, -1): 
        x[i,:] = (b[i,:] - np.dot(U[i,i+1:], x[i+1:,:]))/U[i,i]

    return x


def householder_solve(A, b):
    """
    Given a real mxm matrix A, use the Householder transformation to solve
    Ax_i=b_i, i=1,2,...,k.

    :param A: an mxm-dimensional numpy array
    :param b: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors b_1,b_2,...,b_k.

    :return x: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors x_1,x_2,...,x_k.
    """

    _, n = A.shape
    Ahat = np.column_stack((A, b))
    householder(Ahat, n)
    x = solve_U(Ahat[:,:n], Ahat[:,n:])
    
    return x


def householder_qr(A, complex=False):
    """
    Given a real mxn matrix A, use the Householder transformation to find
    the full QR factorisation of A.

    :param A: an mxn-dimensional numpy array
    :param complex: optional complex array handling

    :return Q: an mxm-dimensional numpy array
    :return R: an mxn-dimensional numpy array
    """

    m, n = A.shape
    Ahat = np.column_stack((A, np.eye(m)))
    householder(Ahat, n, complex=complex)
    Q = Ahat[:,n:].T.conj()
    R = Ahat[:,:n] 

    return Q, R


def householder_ls(A, b):
    """
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """

    _, n = A.shape
    Ahat = np.column_stack((A, b))
    householder(Ahat, n)
    x = solve_U(Ahat[:n,:n], Ahat[:n,n:])[:,0]
    
    return x
