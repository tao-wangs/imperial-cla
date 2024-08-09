import cla_utils
import numpy as np
import matplotlib.pyplot as plt


def tridiag_matrix(m=10):
    """
    Given an integer m, creates an m x m matrix with entries according to specification
    in Coursework 2 Question 1c

    :param m: an integer
    :return A: an mxm dimensional array 
    """

    A = np.zeros([m, m])
    for i in range(m):
        for j in range(m):
            if i == j:
                A[i,j] = 2
            elif i == (j+1) or i == (j-1):
                A[i,j] = -1
            else:
                A[i,j] = 0

    return A


def q1_c():
    """
    Creates the example matrix from 1c and computes its approximate eigenvalues
    using the Pure QR algorithm.

    :return eig: an m-dimensional numpy array 
    """

    A = tridiag_matrix()
    A = cla_utils.pure_QR(A, maxit=50000, tol=1e-12, alt_criteria=True)
    eig = np.diag(A)
    print(f'Approximated eigenvalues\n{eig}')
    return eig


def q1_d():
    """
    Computes the approximate eigenvalues using q1_c() and then uses the analytical 
    formula in Coursework 2 Question 1d to compute the exact eigenvalues. Then, 
    computes the errors in magnitude between the eigenvalues. 
    """

    e = q1_c()
    m = 10
    k = np.arange(1, m+1)
    k = 2 + 2*np.cos((k*np.pi)/(m+1))
    print(f'Actual eigenvalues\n{k}')
    print(f'Error in eigenvalues\n{np.abs(k - e)}')
    print(f'Largest error {np.max(np.abs(k - e))}')
    print(f'Smallest error {np.min(np.abs(k - e))}')

    
def q1_e(A, maxit, tol, return_off_diagonals=False, shift=False):
    """
    For matrix A, apply the practical QR algorithm and return the result.
    1. Reduces a symmetric matrix A to Hessenberg form.
    2. Pure QR until termination, then records the A_{k,k} entry as an eigenvalue.
    3. Deflates the matrix A by 1 dimension, from k x k to (k-1) x (k-1). 

    :param A: an mxm numpy array
    :param maxit: the maximum number of iterations
    :param tol: termination tolerance
    :param return_off_diagonals: default False 
    :param shift: optionally does shifted QR, default False 

    :return Ak: the result
    :return off_diags: the concatenated array of off diagonal A_{k, k-1} values. 
    """

    m = A.shape[0]
    cla_utils.hessenberg(A)
    T = A.copy()
    eigs = np.array([])
    
    if return_off_diagonals:
        off_diags = np.array([])

    if shift:
        Q, R = cla_utils.householder_qr(T)
        T = R @ Q
    
    for k in range(m, 1, -1):
        if shift: 
            if return_off_diagonals: 
                T, off_diags_k = cla_utils.pure_QR(T, maxit, tol, alt_criteria=True, return_off_diagonals=True, shift=True)
                off_diags = np.concatenate((off_diags, off_diags_k))
            else:
                T= cla_utils.pure_QR(T, maxit, tol, alt_criteria=True, shift=True)
        else:
            if return_off_diagonals: 
                T, off_diags_k = cla_utils.pure_QR(T, maxit, tol, alt_criteria=True, return_off_diagonals=True)
                off_diags = np.concatenate((off_diags, off_diags_k))
            else: 
                T = cla_utils.pure_QR(T, maxit, tol, alt_criteria=True)

        
        eigs = np.append(eigs, T[k-1, k-1])    
        T = T[:k-1, :k-1]

    eigs = np.append(eigs, T[0, 0])

    if return_off_diagonals:
        return eigs, off_diags
    else:
        return eigs


def q1_f(A):
    """
    Given a matrix A, calls our practical QR algorithm implemented in q1_e() returning
    the concatenated off diagonal entries, then plots the array against array index. 

    :param A: an mxm numpy array
    """

    _, off_diags = q1_e(A, maxit=50000, tol=1e-12, return_off_diagonals=True)

    plt.plot(np.arange(len(off_diags)), off_diags)
    plt.title('Non-shifted QR Algorithm')
    plt.yscale('log')
    plt.xlabel('Array index')
    plt.ylabel('Off-diagonal value (k, k-1)')
    plt.grid(True)
    plt.show()


def q1_g(A): 
    """
    Given a matrix A, calls our practical QR algorithm implemented in q1_e() with shift, 
    returning the concatenated off diagonal entries, then plots the array against array index. 

    :param A: an mxm numpy array
    """

    _, off_diags = q1_e(A, maxit=50000, tol=1e-12, return_off_diagonals=True, shift=True)

    plt.plot(np.arange(len(off_diags)), off_diags)
    plt.title('Shifted QR Algorithm')
    plt.yscale('log')
    plt.xlabel('Array index')
    plt.ylabel('Off-diagonal value (k, k-1)')
    plt.grid(True)
    plt.show()


def hilbert_matrix(m=10):
    """
    Given an integer m, creates an m x m matrix with entries according to specification
    in Coursework 2 Question 1h

    :param m: an integer
    :return A: an mxm dimensional array 
    """

    A = np.zeros([m, m])
    for i in range(m):
        for j in range(m):
            A[i, j] = 1.0 / (i + j + 1)

    return A


def q1_h():
    """
    Creates the example matrices from Question 1c and Question 1h, then compares
    the required number of iterations for both for modified QR algorithm both with and without shifts.
    """

    A_tridiag = tridiag_matrix()
    A_hilbert = hilbert_matrix()

    # Compare without shift
    q1_f(A_tridiag.copy())
    q1_f(A_hilbert.copy())

    #Compare with shift
    q1_g(A_tridiag.copy())
    q1_g(A_hilbert.copy())
        