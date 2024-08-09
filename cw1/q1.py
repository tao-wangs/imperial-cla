import cla_utils
import numpy as np


def GS_classical_ls(A, b):
    """
    Given a mxn matrix A and an m dimensional vector b, 
    finds the least squares solution to Ax = b using classical Gram-Schmidt algorithm.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """

    _, n = A.shape
    Ahat = np.column_stack((A, b))
    R = cla_utils.GS_classical(Ahat)
    x = cla_utils.solve_U(R[:n,:n], R[:n,n:])[:,0]

    return x


def GS_modified_ls(A, b):
    """
    Given a mxn matrix A and an m dimensional vector b, 
    finds the least squares solution to Ax = b using modified Gram-Schmidt algorithm.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """

    _, n = A.shape
    Ahat = np.column_stack((A, b))
    R = cla_utils.GS_modified(Ahat)
    x = cla_utils.solve_U(R[:n,:n], R[:n,n:])[:,0]

    return x


#Investigation of least squares fit problem. 
def compare_QR_algorithms(A0, y, c0):
    """
    Given a mxn Vandermonde matrix A, an m dimensional vector y and an n dimensional coefficient array 
    c0, finds the least squares solution to Ax = b using each QR algorithm and compares errors. 

    :param A: an mxn dimensional numpy array 
    :param y: an m-dimensional numpy array 
    :param c0: an n-dimensional numpy array 
    """

    # Perform least squares with each QR algorithm and finds error magnitude and residual
    A1 = A0.copy()
    coefficients = GS_classical_ls(A1, y)
    print(f'GS_classical QR results:')
    print(f'Provided Coefficients {c0}')
    print(f'Calculated Coefficients {coefficients}')
    print(f'Coefficient Error: {np.sum(np.abs(coefficients - c0))}')
    print(f'Residual: {np.linalg.norm(np.dot(A0.T, np.dot(A0, coefficients) - y))}\n')

    A2 = A0.copy()
    coefficients = GS_modified_ls(A2, y)
    print('GS_modified QR results:')
    print(f'Provided Coefficients {c0}')
    print(f'Calculated coefficients {coefficients}')
    print(f'Coefficient error: {np.sum(np.abs(coefficients - c0))}')
    print(f'Residual: {np.linalg.norm(np.dot(A0.T, np.dot(A0, coefficients) - y))}\n')

    A3 = A0.copy()
    coefficients = cla_utils.householder_ls(A3, y)
    print('Householder QR results:')
    print(f'Provided Coefficients {c0}')
    print(f'Calculated coefficients {coefficients}')
    print(f'Coefficient error: {np.sum(np.abs(coefficients - c0))}')
    print(f'Residual: {np.linalg.norm(np.dot(A0.T, np.dot(A0, coefficients) - y))}\n')
    