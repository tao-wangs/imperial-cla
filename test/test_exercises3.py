'''Tests for the third exercise set.'''
import pytest
import cla_utils
from numpy import random
import numpy as np


@pytest.mark.parametrize('m', [20, 40, 87])
def test_householder(m):
    random.seed(1878*m)
    A = random.randn(m, m)
    A0 = 1.0*A  # make a deep copy
    status = cla_utils.householder(A0)
    R = A0
    assert(status == None)
    assert(np.allclose(R, np.triu(R)))  # check R is upper triangular
    assert(cla_utils.norm(np.dot(R.T, R) - np.dot(A.T, A)) < 1.0e-6)


@pytest.mark.parametrize('m, k', [(20, 4), (204, 100), (18, 7)])
def test_solve_U(m, k):
    random.seed(1002*m + 2987*k)
    b = random.randn(m, k)
    _, U = cla_utils.householder_qr(random.randn(m,m))
    #check that the solver works
    x = cla_utils.solve_U(U, b)
    err1 = b - np.dot(U, x)
    assert(cla_utils.norm(err1) < 1.0e-6)
    #check that an upper triangular solver is being used
    A = random.randn(m, m)
    err2 = b - np.dot(A, x)
    assert(cla_utils.norm(err2) > 1.0e-6)

    
@pytest.mark.parametrize('m, n', [(20, 7), (40, 13), (87, 9)])
def test_householder_solve(m, n):
    random.seed(2432*m + 7438*n)
    A = random.randn(m, m)
    x0 = random.randn(m, n)
    b = np.dot(A, x0)
    x = cla_utils.householder_solve(A, b)
    assert(cla_utils.norm(x - x0) < 1.0e-6)


@pytest.mark.parametrize('m, n', [(20, 7), (40, 13), (87, 9)])
def test_householder_qr(m, n):
    random.seed(4732*m + 1238*n)
    A = random.randn(m, n)
    A0 = 1*A
    Q, R = cla_utils.householder_qr(A0)

    # check orthonormality
    assert(cla_utils.norm(np.dot(np.conj(Q.T), Q) - np.eye(m)) < 1.0e-6)
    # check upper triangular
    assert(np.allclose(R, np.triu(R)))
    # check QR factorisation
    assert(cla_utils.norm(np.dot(Q, R) - A) < 1.0e-6)


@pytest.mark.parametrize('m, n', [(3, 2), (20, 7), (40, 13), (87, 9)])
def test_householder_ls(m, n):
    random.seed(8473*m + 9283*n)
    A = random.randn(m, n)
    b = random.randn(m)

    A0 = A.copy()
    x = cla_utils.householder_ls(A, b)
    #!!!change test param to b

    #check normal equation residual
    assert(cla_utils.norm(np.dot(A0.T, np.dot(A0, x) - b)) < 1.0e-6)


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
