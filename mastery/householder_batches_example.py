import mastery
import numpy as np

m, n = 4, 3
np.random.seed(0)
A = np.random.rand(m, n)
A0 = A.copy()
W_list, Y_list = mastery.householder_qr_batches(A, r=2)
mastery.extend_WY(W_list, Y_list)
Q = mastery.compute_Q(W_list, Y_list)

# Compare original matrix A0 with QR
print(A0 - Q@A) 
