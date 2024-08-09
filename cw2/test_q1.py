'''Tests for Coursework 2 Question 1.'''
import pytest 
import cw2
import numpy as np 


@pytest.mark.parametrize('m', [10])
def test_q_1e(m):
    A = cw2.tridiag_matrix()

    # Analytical formula for exact eigenvalues
    eigs = 2 + 2*np.cos((np.arange(1, m+1)*np.pi)/(m+1))
    eigs = np.sort(eigs)
    
    approx_eigs = cw2.q1_e(A, maxit=50000, tol=1e-12) 
    
    # Check computed difference 
    assert(np.linalg.norm(eigs - approx_eigs) < 1.0e-12)
