# -*- coding: utf-8 -*-

import numpy as np
import regressionData as rg
import time
import pdb

#-------------------
# �N���X�̒�`�n�܂�
class linearRegression():
	#------------------------------------
	# 1) �w�K�f�[�^����у��f���p�����[�^�̏�����
	# x: �w�K���̓f�[�^�i���̓x�N�g���̎������~�f�[�^����numpy.array�j
	# y: �w�K�o�̓f�[�^�i�f�[�^����numpy.array�j
	# kernelType: �J�[�l���̎�ށi������Fgaussian�j
	# kernelParam: �J�[�l���̃n�C�p�[�p�����[�^�i�X�J���[�j
	def __init__(self, x, y, kernelType="linear", kernelParam=1.0):
		# �w�K�f�[�^�̐ݒ�
		self.x = x
		self.y = y
		self.xDim = x.shape[0]
		self.dNum = x.shape[1]
		
		# �J�[�l���̐ݒ�
		self.kernelType = kernelType
		self.kernelParam = kernelParam
	#------------------------------------

	#------------------------------------
	# 2) �ŏ����@��p���ă��f���p�����[�^���œK��
	# �i����̌v�Z��For����p�����ꍇ�j
	def train(self):
		x_ = np.append(self.x, np.ones((1, self.dNum)), axis=0)
		self.w = np.matmul(np.linalg.inv(np.matmul(x_, x_.T)), np.sum([y * x_[:, index] for index, y in enumerate(self.y)], axis=0))

	#------------------------------------

	#------------------------------------
	# 2) �ŏ����@��p���ă��f���p�����[�^���œK���i�s�񉉎Z�ɂ�荂�����j
	def trainMat(self):
		x_ = np.append(self.x, np.ones((1, self.dNum)), axis=0)
		self.w = np.matmul(np.linalg.inv(np.matmul(x_, x_.T)), np.sum(self.y * x_, axis=1))
	#------------------------------------
	
	#------------------------------------
	# 3) �\��
	# x: ���̓f�[�^�i���͎��� x �f�[�^���j
	def predict(self, x):
		x_ = np.append(x, np.ones((1, x.shape[1])), axis=0)
		y = np.matmul(self.w.T, x_)
		return y
	#------------------------------------

	#------------------------------------
	# 4) ��摹���̌v�Z
	# x: ���̓f�[�^�i���͎��� x �f�[�^���j
	# y: �o�̓f�[�^�i�f�[�^���j
	def loss(self, x, y):
		loss = np.sum((y - self.predict(x)) ** 2) / y.size
		return loss
	#------------------------------------

	def calcDist(self, x, z):
		xTile = np.tile(x.reshape(x.shape[1], 1, x.shape[0]), (1, z.shape[1], 1))
		zTile = np.tile(z.reshape(1, z.shape[1], z.shape[0]), (x.shape[1], 1, 1))
		# xTile = np.tile(x.reshape(1, x.shape[1], x.shape[0]), (z.shape[1], 1, 1))
		# zTile = np.tile(z.reshape(z.shape[1], 1, z.shape[0]), (1, x.shape[1], 1))
		dict = np.sqrt(np.sum((xTile - zTile) ** 2, axis=2))
		return dict

	def kernel(self, x):
		K = np.exp(-self.calcDist(self.x, x) / (2 * self.kernelParam**2))
		return K

	def trainMatKernel(self):
		x_ = self.kernel(self.x)
		x_ = np.append(x_, np.ones((1, x_.shape[1])), axis=0)
		x_x_T = np.matmul(x_, x_.T)
		print(x_x_T.shape)
		x_x_T = x_x_T + np.eye(x_x_T.shape[0]) * 0.0001
		self.w = np.matmul(np.linalg.inv(x_x_T), np.sum(self.y * x_, axis=1))

	def predict4Kernel(self, x):
		x_ = self.kernel(x)
		x_ = np.append(x_, np.ones((1, x_.shape[1])), axis=0)
		y = np.matmul(self.w.T, x_)
		return y

	def loss4Kernel(self, x, y):
		loss = np.sum((y - self.predict4Kernel(x)) ** 2) / y.size
		return loss

# �N���X�̒�`�I���
#-------------------

#-------------------
# ���C���̎n�܂�
if __name__ == "__main__":
	
	# 1) �w�K���͎�����2�̏ꍇ�̃f�[�^�[����
	# myData = rg.artificial(200, 100, dataType="1D")
	myData = rg.artificial(200, 100, dataType="1D", isNonlinear=True)
	
	# 2) ���`��A���f��
	# regression = linearRegression(myData.xTrain, myData.yTrain)
	regression = linearRegression(myData.xTrain, myData.yTrain, kernelType="gaussian", kernelParam=1)
	
	# 3) �w�K�iFor���Łj
	sTime = time.time()
	regression.train()
	eTime = time.time()
	print("train with for-loop: time={0:.4} sec".format(eTime-sTime))
	
	# 4) �w�K�i�s��Łj
	sTime = time.time()
	regression.trainMat()
	eTime = time.time()
	print("train with matrix: time={0:.4} sec".format(eTime-sTime))

	sTime = time.time()
	regression.trainMatKernel()
	eTime = time.time()
	print("train with kernel: time={0:.4} sec".format(eTime - sTime))

	# 5) �w�K�������f����p���ė\��
	print("loss={0:.3}".format(regression.loss4Kernel(myData.xTest, myData.yTest)))

	# 6) �w�K�E�]���f�[�^����ї\�����ʂ��v���b�g
	predict = regression.predict4Kernel(myData.xTest)
	myData.plot(predict,isTrainPlot=False)
	
#���C���̏I���
#-------------------
	