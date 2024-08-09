import cw1 
import numpy as np

# Case 1: Near linear dependence between columns  
print("Comparing QR Algorithms where A has nearly linearly dependent columns\n")
x = np.arange(1, 10, 0.1)
c0 = np.poly1d([3, -2, 0.5, 0.2]) # 3 * x**3 - 2 * x**2 + 0.5 * x + 0.2  
y = c0(x)
A0 = np.vander(x, 4, increasing=True)
print("Condition number of A:", np.linalg.cond(A0))
cw1.compare_QR_algorithms(A0, y, np.flip(c0))


# Case 2: Large matrix with same number of rows, but more columns. m>n condition still holds.
print("\nComparing QR Algorithms where A has nearly linearly dependent columns, but A is larger (m>n still holds)\n")
c0 = np.poly1d([0.5, -1.5, 0.3, 1, 3, -2, 0.5, 0.2]) # 7-degree polynomial 
A0 = np.vander(x, 8, increasing=True)
y = c0(x)
print("Condition number of A:", np.linalg.cond(A0))
cw1.compare_QR_algorithms(A0, y, np.flip(c0))


# Case 3: Large off-diagonal values 
print("\nComparing QR Algorithms where A has large off-diagonal values\n")
x = np.arange(1, 20)
c0 = np.poly1d([3, -2, 0.5, 0.2]) # 3 * x**3 - 2 * x**2 + 0.5 * x + 0.2  
y = c0(x)
A0 = np.vander(x, 4, increasing=True)
print("Condition number of A:", np.linalg.cond(A0))
cw1.compare_QR_algorithms(A0, y, np.flip(c0))


# Case 4: High scaling between columns  
print("\nComparing QR Algorithms where A has large scaling between columns\n")
x = np.arange(10, 100, 10) 
c0 = np.poly1d([6, 3, -2, 0.5, 0.2]) # 6 * x**4 + 3 * x**3 - 2 * x**2 + 0.5 * x + 0.2  
y = c0(x)
A0 = np.vander(x, 5, increasing=True)
print("Condition number of A:", np.linalg.cond(A0))
cw1.compare_QR_algorithms(A0, y, np.flip(c0))
