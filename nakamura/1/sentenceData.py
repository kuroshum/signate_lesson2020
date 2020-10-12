# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#-------------------
# クラスの定義始まり
class sentence:
	dataPath = '/Users/A/intelligentSystemTraining/sentiment_labelled_sentences'    # データのフォルダ名

	#------------------------------------
	# CSVファイルの読み込み
	# fname: ファイルパス（文字列）
	def __init__(self,fname):

		# ファイルのパス設定
		fullpath = os.path.join(self.dataPath,fname)

		# csv形式のデータ読み込み
		self.data = pd.read_csv(fullpath,'\t')
	#------------------------------------

	#------------------------------------
	# 文字列検索
	# keyword: 検索キーワード（文字列）
	def search(self, keyword):
		# sentence列で、keywordを含む要素のインデックスを取得
		results = self.data['sentence'].str.contains(keyword)

		# np.arrayとして返す
		return self.data['sentence'][results].values
	#------------------------------------

	def getPositiveSentence(self):
		TFarr = self.data['score']==1
		#print(TFarr)
		result = np.array([])

		for ind in np.arange(len(TFarr)):
			if TFarr[ind]==True:
				result = np.append(result,self.data['sentence'][ind])

		return result

	def plotScoreRatio(self,keyword):
		results = self.data['sentence'].str.contains(keyword)
		scoreall = self.data['score'][results].values
		score0=0
		score1=0
		for ind in np.arange(len(scoreall)):
			if scoreall[ind]==0:
				score0+=1
			else:
				score1+=1

		left = np.array([1,2])
		height = [score0/len(scoreall), score1/len(scoreall)]
		label = ["negative","positive"]
		plt.bar(left,height,tick_label=label,align="center")
		plt.title('keyword:{}'.format(keyword))
		plt.xlabel("Score")
		plt.ylabel("Ratio")
		plt.show()

# クラスの定義終わり
#-------------------

#-------------------
# メインの始まり
if __name__ == "__main__":
	# データファイルamazon_cells_labelled.txtを指定して、インスタンス化
	myData = sentence("amazon_cells_labelled.txt")

	# 検索
	#results = myData.search("very good")

	# 検索結果の表示
	#for ind in np.arange(len(results)):
		#print(ind,":",results[ind])

	review1 = myData.getPositiveSentence()
	for ind in np.arange(len(review1)):
		print(ind,":",review1[ind])

	myData.plotScoreRatio("not")
#メインの終わり
#-------------------
