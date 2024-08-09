import numpy as np
import numpy.random as random

from .exercises3 import householder_solve, householder_qr
from .exercises8 import hessenberg

def get_A100():
    """
    Return A100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    return A


def get_B100():
    """
    Return B100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A[np.tril_indices(m, -2)] = 0
    return A


def get_C100():
    """
    Return C100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    return A


def get_D100():
    """
    Return D100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    A[np.tril_indices(m, -2)] = 0
    A[np.triu_indices(m, 2)] = 0
    return A


def get_A3():
    """
    Return A3 matrix for investigating power iteration.
    
    :return A3: a 3x3 numpy array.
    """

    return np.array([[ 0.76505141, -0.03865876,  0.42107996],
                     [-0.03865876,  0.20264378, -0.02824925],
                     [ 0.42107996, -0.02824925,  0.23330481]])


def get_B3():
    """
    Return B3 matrix for investigating power iteration.

    :return B3: a 3x3 numpy array.
    """
    return np.array([[ 0.76861909,  0.01464606,  0.42118629],
                     [ 0.01464606,  0.99907192, -0.02666057],
                     [ 0.42118629, -0.02666057,  0.23330798]])


def pow_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a matrix A, apply the power iteration algorithm with initial
    guess x0, until either 

    ||r|| < tol where

    r = Ax - lambda*x,

    or the number of iterations exceeds maxit.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of power iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing all \
    the iterates.
    :return lambda0: the final eigenvalue.
    """ 

    vk = x0
    m = A.shape[0]
    lambda0 = None
    iterates = np.zeros([m, maxit])

    for k in range(1, maxit):
        iterates[:, k-1] = vk
        w = np.dot(A, vk)
        vk = w / np.linalg.norm(w)
        lambda0 = np.inner(vk, A @ vk)

        # Check if eigenvalue is below tolerance 
        if np.linalg.norm(A @ vk - lambda0*vk) < tol:
            break
    
    x = iterates if store_iterations else vk

    return x, lambda0


def inverse_it(A, x0, mu, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the inverse iteration algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param mu: a floating point number, the shift parameter
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, a maxit dimensional numpy array containing \
    all the iterates.
    """
    
    v_iterates = np.zeros([m, maxit])
    l_iterates = np.zeros(maxit)

    m = A.shape[0]
    I = np.eye(m)
    vk = x0

    for k in range(1, maxit):
        v_iterates[:, k-1] = vk 
        w = householder_solve(A - mu*I, vk).flatten()
        vk = w / np.linalg.norm(w)
        l = np.inner(vk, np.dot(A, vk))
        l_iterates[k-1] = l
        
        # Check if eigenvalue is below tolerance 
        if np.linalg.norm(np.dot(A, vk) - l*vk) < tol:
            break
    
    if store_iterations:
        return v_iterates, l_iterates
    else:
        return vk, l
    

def rq_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the Rayleigh quotient algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, an m dimensional numpy array containing \
    all the iterates.
    """

    m = A.shape[0]
    v_iterates = np.zeros([m, maxit])
    l_iterates = np.zeros(maxit)
    vk = x0 
    lk = np.inner(vk, np.dot(A, vk))
    I = np.eye(m)

    for k in range(1, maxit):
        v_iterates[:, k-1] = vk
        l_iterates[k-1] = lk 
        w = householder_solve(A - lk*I, vk).flatten()
        vk = w / np.linalg.norm(w)
        lk = np.inner(vk, np.dot(A, vk))

        # Check if eigenvalue is below tolerance 
        if np.linalg.norm(A@vk - lk*vk) < tol:
            break
    
    if store_iterations:
        return v_iterates, l_iterates
    else:
        return vk, lk


def pure_QR(A, maxit, tol, alt_criteria=False, return_off_diagonals=False, shift=False, shift_value=None):
    """
    For matrix A, apply the QR algorithm and return the result.
    :param A: an mxm numpy array
    :param maxit: the maximum number of iterations
    :param tol: termination tolerance
    :param return_off_diagonals: default False 
    :param shift: optionally does shifted QR, default False 
    :param shift_value: optional shift value from the user, default None

    :return Ak: the result
    :return off_diags: the concatenated array of off diagonal A_{k, k-1} values. 
    """

    m = A.shape[0]
    Ak = A
    k = 0 
    I = np.eye(m, dtype=A.dtype)
    
    mu = None 
    off_diagonals = None 
    
    if return_off_diagonals:
        off_diagonals = np.array([], dtype=A.dtype)
    
    while k < maxit:
        if shift:
            if shift_value:
                mu = shift_value
            else:
                mu = Ak[m-1, m-1] 
            Ak -= mu * I           

        Qk, Rk = householder_qr(Ak, complex=True) if A.dtype == np.complex128 else householder_qr(Ak)
        Ak = Rk @ Qk 
        
        if shift:
            Ak += mu * I 

        if return_off_diagonals:
            off_diagonals = np.append(off_diagonals, np.abs(Ak[m-1, m-2]))

        if alt_criteria and np.linalg.norm(Ak[m-1, m-2]) < tol:
            break 
        
        elif np.linalg.norm(Ak[np.tril_indices(m, -1)])/m**2 < tol:
            break 

        k += 1

    if return_off_diagonals:
        return Ak, off_diagonals
    else:
        return Ak    
