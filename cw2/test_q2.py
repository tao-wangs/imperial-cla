'''Tests for Coursework 2 Question 2.'''
import pytest 
import cw2
from numpy import random
import numpy as np 


@pytest.mark.parametrize('m', [10])
def test_q_2b(m):
    b = random.randn(m)
    A = cw2.tridiag_matrix(m)
    Ahat = np.triu(A)
    y = cw2.GMRES_preconditioner(b.reshape(-1, 1))[:,0]
    
    # Check computed difference 
    assert(np.linalg.norm(Ahat @ y - b) < 1.0e-12)
