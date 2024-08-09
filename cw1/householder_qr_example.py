import cw1
import numpy as np

m, n = 4, 3
np.random.seed(0)
A = np.random.rand(m, n)
A0 = A.copy()
vs, betas = cw1.householder_qr_extension(A, lists=True)
Wn, Yn = cw1.householder_qr_get_WY(vs, betas)
Q = np.eye(m) - np.dot(Wn, Yn.T)

# Compare original matrix A0 with QR
print(A0 - Q@A) 
