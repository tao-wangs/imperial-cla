import numpy as np
import cla_utils


def Q1AQ1s(A):
    """
    For a matrix A, find the unitary matrix Q1 such that the first
    column of Q1*A has zeros below the diagonal. Then return A1 = Q1*A*Q1^*.

    :param A: an mxm numpy array

    :return A1: an mxm numpy array
    """

    sign = lambda x: -1 if x < 0 else 1

    x1 = A[:, 0].copy()
    x1[0] += sign(x1[0])*np.linalg.norm(x1)
    x1 /= np.linalg.norm(x1)
    A -= 2*np.outer(x1, np.dot(x1.conj(), A))
    A1 = A.T - 2*np.outer(x1, np.dot(x1.conj(), A.T))

    return A1


def hessenberg(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place.

    :param A: an mxm numpy array
    """

    sign = lambda x: -1 if x < 0 else 1

    m, _ = A.shape
    
    for k in range(m-2):
        vk = A[k+1:, k].copy()
        vk[0] += sign(vk[0])*np.linalg.norm(vk)
        vk /= np.linalg.norm(vk)
        A[k+1:, k:] -= 2*np.outer(vk, np.dot(vk.conj().T, A[k+1:, k:])) 
        A[:, k+1:] -= 2*np.outer(np.dot(A[:, k+1:], vk), vk.conj().T) 
    

def hessenbergQ(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place, and return the matrix Q
    for which QHQ^* = A.

    :param A: an mxm numpy array
    
    :return Q: an mxm numpy array
    """

    sign = lambda x: -1 if x < 0 else 1

    m, _ = A.shape

    Q = np.eye(m)
    
    for k in range(m-2):
        vk = A[k+1:, k].copy()
        vk[0] += sign(vk[0])*np.linalg.norm(vk)
        vk /= np.linalg.norm(vk)
        A[k+1:, k:] -= 2*np.outer(vk, np.dot(vk.conj(), A[k+1:, k:])) 
        Q[:, k+1:] -= 2*np.outer(np.dot(Q[:, k+1:], vk), vk.conj())
        A[:, k+1:] -= 2*np.outer(np.dot(A[:, k+1:], vk), vk.conj()) 
    
    return Q


def hessenberg_ev(H):
    """
    Given a Hessenberg matrix, return the eigenvectors.

    :param H: an mxm numpy array

    :return V: an mxm numpy array whose columns are the eigenvectors of H

    Do not change this function.
    """
    m, n = H.shape
    assert(m==n)
    assert(cla_utils.norm(H[np.tril_indices(m, -2)]) < 1.0e-6)
    _, V = np.linalg.eig(H)
    return V


def ev(A):
    """
    Given a matrix A, return the eigenvectors of A. This should
    be done by using your functions to reduce to upper Hessenberg
    form, before calling hessenberg_ev (which you should not edit!).

    :param A: an mxm numpy array

    :return V: an mxm numpy array whose columns are the eigenvectors of A
    """

    Q = hessenbergQ(A)
    V = hessenberg_ev(A)

    return Q @ V
