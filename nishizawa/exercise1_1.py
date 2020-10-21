# -*- coding: utf-8 -*-

# numpyをnpとしてインポート
import numpy as np

# pandasをpdとしてインポート
import pandas as pd

# リストの作成
d_list1 = ['Yamada Taro', 'Yamada Hanako', 'Wakayama Dai']
d_list2 = [172.5, 160.5, 180.2]
d_list3 = ['name','height']

# リストからnumpy arrayへの変換
d_array1 = np.array(d_list1)
d_array2 = np.array(d_list2)
d_array3 = np.array(d_list3)


# numpy arrayからdataframeへの変換
# pandas.Seriesを用いて項目名にd_array3の要素（name, height）を設定する
column1 = pd.Series(d_array1,name=d_array3[0])
column2 = pd.Series(d_array2,name=d_array3[1])

# ２つの列をpandas.concatで結合し、dataframeを作成する
d_dataframe = pd.concat([column1,column2],axis=1)
print("d_dataframe:\n",d_dataframe,"\n")



#演習1　年齢の列の追加
d_dataframe['age'] =[21, 19, 30] 
print("d_dataframe:\n",d_dataframe,"\n")