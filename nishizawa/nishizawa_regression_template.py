# -*- coding: utf-8 -*-

import numpy as np
import regressionData as rg
import time
import pdb

#-------------------
# クラスの定義の始まり
class linearRegression():
    #------------------------------------
    # 1) 学習データおよびモデルパラメータの初期化
    # x:学習入力データ(入力ベクトルの次元数*データ数のnumpy.array)
    # y:学習出力データ(データ数のnumpy.array)
    # kernelType: カーネルの種類(文字列:gaussian)
    # kernelParam: カーネルのハイパーパラメータ(スカラー)
    def __init__(self, x, y, kernelType="linear", kernelParam=1.0):
        # 学習データの設定
        self.x = x
        self.y = y
        self.xDim = x.shape[0]
        self.dNum = x.shape[1]
        
        # カーネルの設定
        self.kernelType = kernelType
        self.kernelParam = kernelParam
    #------------------------------------

    #------------------------------------
    # 2) 最小二乗法を用いてモデルパラメータを最適化
    # (分子・分母の計算にFor分を用いた場合)
    def train(self):
        self.w = np.zeros([self.xDim,1])
    #------------------------------------

    #------------------------------------
    # 2) 最小二乗法を用いてモデルパラメータを最適化
    # (分子・分母の計算に行列計算を用いた場合)
    def trainMat(self):
        array1  = np.ones((1,self.dNum))
        x1 = np.vstack((self.x,array1))
        A = np.linalg.inv(np.matmul(x1,x1.T))
        B = np.sum(self.y*x1,axis=1)
        self.w = np.matmul(A,B)
    #------------------------------------
    
    #------------------------------------
    # 3) 予測
    # x: 入力データ(入力次元 * データ数)
    def predict(self,x):
        if self.kernelType == "linear":
            x1 = np.vstack((x,np.ones((1,x.shape[1]))))
            y= self.w@x1
            return y
        else:
            k1 = self.kernel(x)
            array1  = np.ones((1,k1.shape[0]))
            x1 = np.vstack((k1,np.ones((1,x.shape[1]))))
            return self.w@x1
    #------------------------------------

    #------------------------------------
    # 4) 二乗損失の計算
    # x: 入力データ(入力次元 * データ数)
    # y: 出力データ(データ数)
    def loss(self,x,y):
        array1  = np.ones((1,x.shape[1]))
        x1 = np.vstack((x,array1))
        loss = sum((y - (self.w@x1))**2)/y.shape[0]
        return loss
    #------------------------------------

    #------------------------------------
    #5)カーネルの計算
    #x:カーネルを計算する対象の行列(次元*データ数)
    def kernel(self,x):
        #(self.xの各データ点xiと行列xの各データ点xjと間のカーネル値k(xi,xj)を各要素に持つグラム行列を計算)
        K = np.exp(-(self.calcDist(self.x,x)**2)/(2*(self.kernelParam**2)))
        return K
    #------------------------------------

    #------------------------------------
    # 6)2つのデータ集合管のすべての組み合わせの距離の計算
    #x:行列(次元*データ数)
    #z:行列(次元*データ数)
    def calcDist(self,x,z):
    #(行列xのデータ点x1,x2,...,xNと、行列zのデータ点z1,z2,...,zMとの間のM*N個の距離を計算)
        x1=np.tile(x,(z.shape[1],1,1)).transpose(2,0,1)
        z1=np.tile(z.T,(x.shape[1],1,1))
        dist=(x1-z1)**2
        dist=np.sqrt(np.sum(dist,2))
        return dist
    #------------------------------------
    def trainMatKernel(self):
        ##gaussian時、wの更新
        if self.kernelType != "linear":
            #xtrainとxtrainのカーネル (200,200)
            K = self.kernel(self.x)
            array1 = np.ones((1,K.shape[1]))
            K1 = np.hstack((K,array1.T))
            A=(K1.T@K1)
            E = np.identity(A.shape[0])
            A = A+(E*0.1)
            a=(np.linalg.inv(A))@(np.sum((self.y*K1.T),1))
            self.w = a
    #------------------------------------
# クラスの定義の終わり
#-------------------

#-------------------
# メインの始まり
if __name__ == "__main__":
    
    # 1) 学習入力次元が2の場合
    #myData = rg.artificial(200,100, dataType="1D")
    myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)
    
    # 2) 線形回帰モデル
    #regression = linearRegression(myData.xTrain,myData.yTrain)
    regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=1)
    
    # 3) 学習(For文版)
    sTime = time.time()
    regression.train()
    eTime = time.time()
    print("train with for-loop: time={0:.4} sec".format(eTime-sTime))
    
    # 4) 学習(行列版)
    sTime = time.time()
    regression.trainMat()
    eTime = time.time()
    print("train with matrix: time={0:.4} sec".format(eTime-sTime))

    # 5) 学習したモデルを用いて予測
    print("loss={0:.3}".format(regression.loss(myData.xTest,myData.yTest)))
    regression.trainMatKernel()

    # 6) 学習・評価データ及び予測結果をプロット
    predict = regression.predict(myData.xTest)
    
    myData.plot(predict,isTrainPlot=False)
    
#メインの終わり
#-------------------