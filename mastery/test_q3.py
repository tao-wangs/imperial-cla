'''Tests for coursework mastery question.'''
import pytest 
import mastery
from numpy import random
import numpy as np 


@pytest.mark.parametrize('m, n', [(3, 2), (20, 7), (40, 13), (87, 9)]) 
def test_householder_qr(m, n):
    random.seed(8473*m + 9283*n)
    A = random.randn(m, n)
    r = random.randint(1, n)

    A0 = A.copy()
    W_list, Y_list = mastery.householder_qr_batches(A, r)
    mastery.extend_WY(W_list, Y_list)
    Q = mastery.compute_Q(W_list, Y_list)

    # Compute error
    assert(np.linalg.norm(np.dot(Q, A) - A0) < 1.0e-12)
