import cla_utils
import numpy as np 


def q3_matrix(m=10):
    """
    Given an integer m, creates an m x m matrix with entries according to specification
    in Coursework 2 Question 3a 

    :param m: an integer
    :return A: an mxm dimensional array 
    """

    A = np.zeros([m, m])
    for i in range(m):
        for j in range(m):
            if i == j+1:
                A[i, j] = 1 
            elif i == j-1: 
                A[i, j] = -1
            else:
                A[i, j] = 0
                
    return A


def q3_a():
    """
    Applies the pure QR algorithm to the matrix defined in Question 3a and investigates
    the matrix structure and approximated eigenvalues in the limit of a large number of iterations.  
    """

    A = q3_matrix()
    # Compute approximate eigenvalues using pure QR
    A = cla_utils.pure_QR(A, maxit=50000, tol=1e-5)
    print(f'APPOXIMATED eigenvalues are {np.diag(A)}')

    print(f'Matrix structure after applying Pure QR: ')
    print(A)
    