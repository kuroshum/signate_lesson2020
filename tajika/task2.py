import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class sentence:
	dataPath = 'sentiment_labelled_sentences'    # データのフォルダ名
	
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
	def getPositiveSentence(self):
		is1stlst=self.data['score']==1
		results=[ind for ind in np.arange(len(self.data)) if is1stlst[ind]]
		return self.data['sentence'][results].values
	def plotScoreRatio(self,keyword):
		hasthewordlst=self.data["sentence"].str.contains(keyword)
		intersected=hasthewordlst & self.data['score']==1
		posrate=np.count_nonzero(intersected)/len(self.data)
		negarate=1-posrate
		plt.bar((0,1),(negarate,posrate),tick_label=("negative","positive"),align="center")


if __name__ == "__main__":
	# データファイルamazon_cells_labelled.txtを指定して、インスタンス化
	myData = sentence("amazon_cells_labelled.txt")

	# 検索
	# results = myData.search("very good")
	
	# 検索結果の表示
	# for ind in np.arange(len(results)):
	# 	print(ind,":",results[ind])
	print(myData.getPositiveSentence())
	myData.plotScoreRatio("not")
	plt.savefig("t2.png")
