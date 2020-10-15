# coding: utf-8

import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt


# -------------------
# クラスの定義始まり
class Sentence:
	dataPath = ''  # データのフォルダ名

	# ------------------------------------
	# CSVファイルの読み込み
	# fname: ファイルパス（文字列）
	def __init__(self, fname):
		# ファイルのパス設定
		fullpath = os.path.join(self.dataPath, fname)

		# csv形式のデータ読み込み
		self.data = pd.read_csv(fullpath, '\t')

	# ------------------------------------

	# ------------------------------------
	# 文字列検索
	# keyword: 検索キーワード（文字列）
	def search(self, keyword):
		# sentence列で、keywordを含む要素のインデックスを取得
		results = self.data['sentence'].str.contains(keyword)

		# np.arrayとして返す
		return self.data['sentence'][results].values

	# 宿題2-1
	def get_positive_sentence(self):
		return self.data[self.data['score'] == 1]['sentence'].values

	# 宿題2-2
	def plot_score_ratio(self, keyword):
		results = self.data[self.data['sentence'].str.contains(keyword)]
		ratio = [
			len(results[results['score'] == 0]),
			len(results[results['score'] == 1])
		]
		plt.bar(['negative', 'positive'], ratio)
		plt.title(f"keyword: {keyword}")
		plt.xlabel("score")
		plt.ylabel("ratio")
		plt.savefig("scoreRatio.png")
		plt.show()


# ------------------------------------

# クラスの定義終わり
# -------------------

# -------------------
# メインの始まり
if __name__ == "__main__":
	# データファイルamazon_cells_labelled.txtを指定して、インスタンス化
	myData = Sentence("amazon_cells_labelled.txt")

	# 検索
	results = myData.search("very good")

	# 宿題2-1
	results = myData.get_positive_sentence()

	# 宿題2-2
	results = []
	myData.plot_score_ratio("b")

	# 検索結果の表示
	for ind in np.arange(len(results)):
		print(ind, ":", results[ind])
# メインの終わり
# -------------------
