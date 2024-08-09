import cla_utils
from .q1 import tridiag_matrix
import numpy as np
import matplotlib.pyplot as plt

from .q1 import tridiag_matrix


def GMRES_preconditioner(v):
    """
    Given an m dimensional vector v, uses the upper triangular component of the 
    example matrix defined in Coursework 2 Question 1c as our matrix, then solves 
    upper triangular system of equations. 

    :param v: an m dimensional array 

    :return y: an m dimensional array 
    """

    m = len(v)
    A = tridiag_matrix(m)
    Ahat = np.triu(A)
    y = cla_utils.solve_U(Ahat, v)
    
    return y
    

def q2_c(m=50):
    """
    Compares the number of iterations required for matrices ranging from 1 x 1 to m x m
    to converge using GMRES and preconditioned GMRES, and plots using matplotlib.  

    :param m: an integer specifying the maximum m x m array to compare for both GMRES implementations. 
    """

    nits_pre_array = np.array([])
    nits_array = np.array([])

    for m in range(1, m):
        A = tridiag_matrix(m)
        b = np.random.randn(m)
        b = b / np.linalg.norm(b)

        print(f'EXPERIMENTING WITH SIZE {m}')
    
        _, nits_pre, _, _ = cla_utils.GMRES(A, b, maxit=10000, tol=1e-3, p_function=GMRES_preconditioner, return_nits=True,
                                    return_residuals=True, return_residual_norms=True)
        _, nits, _, _ = cla_utils.GMRES(A, b, maxit=10000, tol=1e-3, return_nits=True,
                                return_residuals=True, return_residual_norms=True)

        print(f'Preconditioned took {nits_pre} iterations')
        print(f'Non-preconditioned took {nits} iterations')

        nits_pre_array = np.append(nits_pre_array, nits_pre)
        nits_array = np.append(nits_array, nits)


    plt.title('GMRES vs Preconditioned-GMRES')
    plt.plot(np.arange(1, m+1), nits_pre_array, label='Preconditioned GMRES')
    plt.plot(np.arange(1, m+1), nits_array, label='GMRES')
    plt.xlabel(f'Matrix size m')
    plt.ylabel(f'Number of iterations')
    plt.legend()
    plt.grid(True)
    plt.show()
