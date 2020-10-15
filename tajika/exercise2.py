import numpy as np
import pdb


A = np.array([ [1,3,2], [-1,0,1], [2,3,0] ])
A_inv = np.linalg.inv(A)
print(f"A_inv:\n{A_inv}\n")

E = np.matmul(A,A_inv)
print(f"product:\n{E}\n")