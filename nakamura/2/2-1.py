import regressionData as rg

myData1 = rg.artificial(200, 100, dataType="1D")
myData2 = rg.artificial(200, 100, dataType="2D")
x1 = myData1.xTrain
x2 = myData2.xTrain

w1 = myData1.train(x1)
print("1次元\n",w1)

w2 = myData2.train(x2)
print("2次元\n",w2)
