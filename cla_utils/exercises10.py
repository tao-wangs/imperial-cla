import numpy as np
import numpy.random as random

from .exercises3 import householder_ls


def arnoldi(A, b, k):
    """
    For a matrix A, apply k iterations of the Arnoldi algorithm,
    using b as the first basis vector.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array, the starting vector
    :param k: integer, the number of iterations

    :return Q: an mx(k+1) dimensional numpy array containing the orthonormal basis
    :return H: a (k+1)xk dimensional numpy array containing the upper \
    Hessenberg matrix
    """

    m = A.shape[0]
    Q = np.zeros((m, k+1), dtype=A.dtype)
    H = np.zeros((k+1, k), dtype=A.dtype)
    Q[:,0] = b / np.linalg.norm(b)

    for n in range(k):
        v = A @ Q[:,n] 
        H[:n+1,n] = np.dot(Q[:,:n+1].T.conj(), v)
        v -= np.dot(Q[:,:n+1], H[:n+1, n])
        H[n+1,n] = np.linalg.norm(v)
        Q[:,n+1] = v / H[n+1,n]

    return Q, H


def GMRES(A, b, maxit, tol, return_residual_norms=False,
          return_residuals=False, p_function=None, return_nits=False):
    """
    For a matrix A, solve Ax=b using the basic GMRES algorithm.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array
    :param maxit: integer, the maximum number of iterations
    :param tol: floating point number, the tolerance for termination
    :param return_residual_norms: logical
    :param return_residuals: logical
    :param p_function: optional preconditioner function

    :return x: an m dimensional numpy array, the solution
    :return nits: if converged, the number of iterations required, otherwise \
    equal to -1
    :return rnorms: nits dimensional numpy array containing the norms of \
    the residuals at each iteration
    :return r: mxnits dimensional numpy array, column k contains residual \
    at iteration k
    """

    m = A.shape[0]
    xn = np.empty((m,0), dtype=A.dtype)
    nits = -1 

    if return_residual_norms:
        rnorms = np.array([])

    if return_residuals:
        r = np.empty((m, 0), dtype=A.dtype)

    if p_function:
        btilda = p_function(b.reshape(-1,1))[:,0]
        btilda_norm = np.linalg.norm(btilda)
        Q = np.zeros([m,1], dtype=A.dtype)
        H = np.empty((1,0), dtype=A.dtype)
        Q[:,0] = btilda / btilda_norm

        for n in range(1, maxit):
            H = np.vstack((H, np.zeros(H.shape[1])))     # Add a row of zeros
            H = np.hstack((H, np.zeros((H.shape[0],1)))) # Add a column of zeros
            nits = n 
            e1_n = np.zeros(n+1)
            e1_n[0] = 1
            v = p_function((A @ Q[:,n-1]).reshape(-1,1))[:,0]
            for j in range(n):
                H[j,n-1] = np.dot(Q[:,j].conj(), v)
                v -= H[j,n-1] * Q[:,j]
            H[n,n-1] = np.linalg.norm(v)
            Q = np.column_stack((Q, v / H[n,n-1]))
            y = householder_ls(H, btilda_norm*e1_n)
            xn = Q[:,:n] @ y 
            res = H @ y - btilda_norm*e1_n
            Rn = np.linalg.norm(res)
            aux = np.zeros(m, dtype=A.dtype)
            aux[:n+1] = res[:m]
            r = np.column_stack((r, aux))
            rnorms = np.append(rnorms, Rn)
            if Rn < tol:
                break 
    else:
        bnorm = np.linalg.norm(b)
        q1 = b / bnorm
        for n in range(1, maxit):
            nits = n
            e1_n = np.zeros(n+1)
            e1_n[0] = 1
            Qn, Hn = arnoldi(A, b, n)
            y = householder_ls(Hn, bnorm*e1_n)
            xn = Qn[:,:n] @ y
            res = Hn @ y - bnorm*e1_n
            Rn = np.linalg.norm(res)
            aux = np.zeros(m, dtype=A.dtype)
            aux[:n+1] = res[:m]
            r = np.column_stack((r, aux))
            rnorms = np.append(rnorms, Rn)
            if Rn < tol:
                break 
    
    if return_nits and return_residuals and return_residual_norms:
        return xn, nits, r, rnorms 
    else:
        return xn, nits
    

def get_AA100():
    """
    Get the AA100 matrix.

    :return A: a 100x100 numpy array used in exercises 10.
    """
    AA100 = np.fromfile('AA100.dat', sep=' ')
    AA100 = AA100.reshape((100, 100))
    return AA100


def get_BB100():
    """
    Get the BB100 matrix.

    :return B: a 100x100 numpy array used in exercises 10.
    """
    BB100 = np.fromfile('BB100.dat', sep=' ')
    BB100 = BB100.reshape((100, 100))
    return BB100


def get_CC100():
    """
    Get the CC100 matrix.

    :return C: a 100x100 numpy array used in exercises 10.
    """
    CC100 = np.fromfile('CC100.dat', sep=' ')
    CC100 = CC100.reshape((100, 100))
    return CC100
