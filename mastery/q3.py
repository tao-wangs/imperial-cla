import cw1
import numpy as np 


def householder_qr_batches(A, r, kmax=None):
    """
    Given a mxn matrix A and integer batch size r, performs householder qr in batches of size r < n.

    :param A: an mxn-dimensional numpy array 
    :param r: an integer representing batch size
    :param kmax: an integer, the number of columns of A to reduce \
    to upper triangular. If not present, will default to n.

    :return W_list, Y_list: a tuple of lists of numpy arrays
    """

    _, n = A.shape
    
    if kmax is None:
        kmax = n

    if r >= kmax:
        raise Exception('Batch size must be less than columns of input matrix') 

    W_list = []
    Y_list = []

    for j in range(0, kmax, r):
        v_list, beta_list = cw1.householder_qr_extension(A[j:,j:j+r], lists=True)
        Wj, Yj = cw1.householder_qr_get_WY(v_list, beta_list)
        A[j:,j+r:] -= np.dot(Yj, np.dot(Wj.T, A[j:,j+r:]))   
        W_list.append(Wj)
        Y_list.append(Yj)
    
    return W_list, Y_list


def extend_WY(W_list, Y_list):
    """
    Given two lists containing W's and Y's respectively, 
    extends their rows with zeros to original row dimension m 
    
    :param W_list: a list of numpy arrays
    :param Y_list: a list of numpy arrays
    """
    
    # Find original number of rows of A
    mmax, _ = W_list[0].shape

    for i in range(1, len(W_list)):
        k, n = W_list[i].shape
        W_list[i] = np.vstack((np.zeros([mmax-k, n]), W_list[i]))
        Y_list[i] = np.vstack((np.zeros([mmax-k, n]), Y_list[i]))


def compute_Q(W_list, Y_list):
    """
    Given two lists containing W's and Y's respectively, computes Q

    :param W_list: a list of numpy arrays
    :param Y_list: a list of numpy arrays

    :return Q: an mxn-dimensional numpy array 
    """

    m, _ = W_list[0].shape
    
    Q = np.eye(m)

    for i in range(0, len(W_list)):
        Q = Q @ (np.eye(m) - np.dot(W_list[i], Y_list[i].T))

    return Q
