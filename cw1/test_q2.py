'''Tests for coursework second question.'''
import pytest 
import cw1
from numpy import random
import numpy as np 


@pytest.mark.parametrize('m, n', [(3, 2), (20, 7), (40, 13), (87, 9)]) 
def test_householder_qr_extension(m, n):
    random.seed(8473*m + 9283*n)
    A = random.randn(m, n)

    A0 = A.copy()
    v_list, beta_list = cw1.householder_qr_extension(A, lists=True)

    Wn, Yn = cw1.householder_qr_get_WY(v_list, beta_list)

    Q = np.eye(m) - np.dot(Wn, Yn.T)

    # Check computed difference 
    assert(np.linalg.norm(np.dot(Q, A) - A0) < 1.0e-12)
