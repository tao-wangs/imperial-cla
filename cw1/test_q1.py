'''Tests for coursework first question.'''
import pytest 
import cla_utils
import cw1
from numpy import random
import numpy as np 


@pytest.mark.parametrize('m, n', [(3, 2), (20, 7)])  # Gentle case for classical GS
def test_GS_classical_ls(m, n):
    random.seed(8473*m + 9283*n)
    x = random.rand(m) + 1j*random.rand(m)
    c0 = np.poly1d(random.rand(n) + 1j*random.rand(n)) 
    A = np.vander(x, n, increasing=True)
    y = c0(x)

    A0 = A.copy()
    c = cw1.GS_classical_ls(A, y)
    
    # Check coefficient error and normal equation residual
    assert(np.linalg.norm(c - np.flip(c0)) < 1.0e-6)
    assert(np.linalg.norm(np.dot(A0.T, np.dot(A0, c) - y)) < 1.0e-6)


@pytest.mark.parametrize('m, n', [(3, 2), (20, 7), (40, 13), (87, 9)]) 
def test_GS_modified_ls(m, n):
    random.seed(8473*m + 9283*n)
    x = random.rand(m) + 1j*random.rand(m)
    c0 = np.poly1d(random.rand(n) + 1j*random.rand(n)) 
    A = np.vander(x, n, increasing=True)
    y = c0(x)

    A0 = A.copy()
    c = cw1.GS_modified_ls(A, y)
    
    # Check coefficient error and normal equation residual
    assert(np.linalg.norm(c - np.flip(c0)) < 1.0e-6)
    assert(np.linalg.norm(np.dot(A0.T, np.dot(A0, c) - y)) < 1.0e-6)


@pytest.mark.parametrize('m, n', [(3, 2), (20, 7), (40, 13), (87, 9)]) 
def test_householder_ls(m, n):
    random.seed(8473*m + 9283*n)
    x = random.rand(m)
    c0 = np.poly1d(random.rand(n))
    A = np.vander(x, n, increasing=True)
    y = c0(x)

    A0 = A.copy()
    c = cla_utils.householder_ls(A, y)
    
    # Check coefficient error and normal equation residual
    assert(np.linalg.norm(c - np.flip(c0)) < 1.0e-6)
    assert(np.linalg.norm(np.dot(A0.T, np.dot(A0, c) - y)) < 1.0e-6)
