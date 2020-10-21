# coding: utf-8

import regressionData as rg
import numpy as np
import pdb

if __name__ == "__main__":
	m = rg.artificial(200, 100, dataType="2D")
	xTrain = np.append(m.xTrain, np.ones((1, m.xTrain.shape[1])), axis=0)
	# w = np.matmul(np.linalg.inv(np.matmul(xTrain, xTrain.T)), np.sum([y * xTrain[:, index] for index, y in enumerate(m.yTrain)], axis=0))
	w = np.matmul(np.linalg.inv(np.matmul(xTrain, xTrain.T)), np.sum(m.yTrain * xTrain, axis=1))
	print(w)
	# pdb.set_trace()
