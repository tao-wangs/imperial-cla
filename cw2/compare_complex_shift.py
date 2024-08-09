from cw2 import q3_matrix
import cla_utils
import numpy as np 

# Compute exact eigenvalues 
A = q3_matrix()
maxit = 50000
tol = 1e-5
complex_shift = 0 + 0.1j

e, _ = np.linalg.eig(A)
print(f'EXACT eigenvalues are {e}')

# Compute approximate eigenvalues using pure QR
A1, off_diags1 = cla_utils.pure_QR(A, maxit, tol, return_off_diagonals=True)

shift_val = 0 + 0.1j
A2, off_diags2 = cla_utils.pure_QR(A + 0j, maxit, tol, return_off_diagonals=True, shift=True, shift_value=complex_shift)
print(f'APPOXIMATED eigenvalues are {np.diag(A2)}')

print(f'Pure QR took {len(off_diags1)} iterations')
print(f'Shifted QR took {len(off_diags2)} iterations')
