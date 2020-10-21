import numpy as np

A = np.array([ [1,3,2], [-1,0,1], [2,3,0] ])
print(f"A:\n{A}\n")

A_inv = np.linalg.inv(A)
print(f"A_inv:\n{A_inv}\n")

A_Ainv = np.matmul(A,A_inv)
print(f"A_A_inv:\n{A_Ainv}\n")
