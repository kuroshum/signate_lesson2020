# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#-------------------
# クラスの定義始まり
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
    #------------------------------------

    #------------------------------------宿題2(1)
    def getPositiveSentence(self):
        #score列が1のインデックスを取得
        results = (self.data['score']==1)
        
        #np.arrayとして返す
        return self.data['sentence'][results].values
    #-------------------------------------
    
    #------------------------------------宿題2(2)
    def plotScoreRatio(self,keyword):
        #表示するデータの初期化
        x1=['nagative','positive']
        y1=[]
        
        # sentence列で、keywordを含む要素のインデックスを取得
        results = self.data['sentence'].str.contains(keyword)
        
        #keywordを含む、score列が0の数を取得
        y1.append(np.count_nonzero(self.data['score'][results]==0))
        #keywordを含む、score列が１の数を取得
        y1.append(np.count_nonzero(self.data['score'][results]==1))
        
        #表の作成
        plt.bar(x1,y1/np.sum(y1))
        
        plt.title("keyword:%s"%keyword,fontsize=14)
        plt.xlabel("Score",fontsize=14)
        plt.ylabel("Ratio",fontsize=14)
        
        #表示
        plt.show()
    #-------------------------------------
# クラスの定義終わり
#-------------------

#-------------------
# メインの始まり
if __name__ == "__main__":
	# データファイルamazon_cells_labelled.txtを指定して、インスタンス化
    myData = sentence("amazon_cells_labelled.txt")

    # 検索
    #results = myData.getPositiveSentence()
    # 検索結果の表示
    #for ind in np.arange(len(results)):
    #   print(ind,":",results[ind])
    
    #keyword:'not'のnp表
    myData.plotScoreRatio('not')

#メインの終わり
#-------------------