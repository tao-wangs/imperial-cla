'''Tests for Coursework 2 Question 3.'''
import pytest
import cla_utils 
from numpy import random
import numpy as np 


@pytest.mark.parametrize('m', [20, 40, 87])
def test_householder(m):
    random.seed(1878*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A0 = 1.0*A  # make a deep copy
    status = cla_utils.householder(A0, complex=True)
    R = A0
    assert(status == None)
    assert(np.allclose(R, np.triu(R)))  # check R is upper triangular
    assert(cla_utils.norm(np.dot(R.T.conj(), R) - np.dot(A.T.conj(), A)) < 1.0e-6)
