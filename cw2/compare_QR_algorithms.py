from cw2 import tridiag_matrix, q1_e
from cla_utils import pure_QR

# Compare number of iterations with unmodified QR algorithm 
A0 = tridiag_matrix()
A1 = A0.copy()
A2 = A0.copy()

_, off_diags1 = q1_e(A1, maxit=50000, tol=1e-12, return_off_diagonals=True)
_, off_diags2 = pure_QR(A2, maxit=50000, tol=1e-12, return_off_diagonals=True)

# Ideally need to also check how close the eigenvalues are to the exact values using 1d) formula. 
print(f'Modified QR Algorithm took {len(off_diags1)} iterations')
print(f'Unmodified QR Algorithm took {len(off_diags2)} iterations')
