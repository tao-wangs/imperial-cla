import cla_utils
import numpy as np


def householder_qr_extension(A, kmax=None, lists=False):
    """
    Given a real mxn matrix A, finds the reduction to upper triangular matrix R
    using Householder transformations. The reduction is done in-place.

    :param A: an mxn-dimensional numpy array
    :param kmax: an integer, the number of columns of A to reduce \
    to upper triangular. If not present, will default to n.
    :param lists: a boolean, the option whether to return the list of v's and beta's \
    used in the Householder transformations. If not present, will default to false.

    :return v_list, beta_list: a tuple of lists with length n, containing v's and beta's
    """
        
    sign = lambda x: -1 if x < 0 else 1

    _, n = A.shape
    if kmax is None:
        kmax = n

    v_list = []
    beta_list = []
    
    for k in range(kmax):
        vk = A[k:, k].copy()
        vk[0] += sign(vk[0])*cla_utils.norm(vk)
        beta = 2/np.dot(vk, vk)
        v_list.append(vk)
        beta_list.append(beta)
        A[k:, k:] -= beta*np.outer(vk, np.dot(vk, A[k:, k:]))
    
    if lists:
        return v_list, beta_list


def householder_qr_get_WY(v_list, beta_list):
    """
    Given an array of v's and beta's, constructs Wn and Yn.

    :param v_list: an array of length n containing v's 
    :param beta_list: an array of length n containing beta's

    :return Wn, Yn: a tuple of mxn-dimensional numpy arrays
    """

    n = len(v_list)
    m = len(v_list[0])

    Wn = np.zeros((m, n))
    Yn = np.zeros((m, n))

    for i in range(n):
        v = v_list[i]
        beta = beta_list[i]
        padding = np.zeros(m - len(v))
        v = np.concatenate((padding, v))
        Wn[:,i] = beta * (v - np.dot(Wn[:,:i], np.dot(Yn[:,:i].T, v))) 
        Yn[:,i] = v

    return Wn, Yn
