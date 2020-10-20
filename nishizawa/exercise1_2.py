import numpy as np
A=np.array([[1,3,2],[-1,0,1],[2,3,0]])

A_inv = np.linalg.inv(A)
print(A_inv)

f_A = np.matmul(A,A_inv)
print(f_A)