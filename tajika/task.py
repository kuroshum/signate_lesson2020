import numpy as np
import pdb

def softmax(a):
	max_a = np.max(a)
	#正規化
	exp_a = np.exp(a-max_a)
	sum_exp=np.sum(exp_a)
	return exp_a/sum_exp

W = np.array([ [1,0,0], [0,1/2,0], [0,0,1/3] ])
H = np.array([ [1,0,0], [0,2,0], [0,0,3] ])
x = np.array([ [1], [2], [3] ])

print(W)
print(softmax(W))
print(H)
print(softmax(H))
print(x)
print(softmax(x))