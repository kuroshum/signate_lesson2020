# -*- coding: utf-8 -*-

# numpyをnpとしてインポート
import numpy as np

# pandasをpdとしてインポート
import pandas as pd

# リストの作成
d_list1 = ['Yamada Taro', 'Yamada Hanako']
d_list2 = [172.5, 160.5]
d_list3 = ['name','height']

# リストからnumpy arrayへの変換
d_array1 = np.array(d_list1)
d_array2 = np.array(d_list2)
d_array3 = np.array(d_list3)

# numpy_arrayの要素の追加
print("要素の追加前：",d_array1,d_array2)
d_array1 = np.append(d_array1,'Wakayama Dai')
d_array2 = np.append(d_array2,180.2)
print("要素の追加後：",d_array1,d_array2)
print("-------\n")

# numpy_arrayの要素への様々な参照方法
print("全ての要素:",d_array1[:])
print("0番目の要素:",d_array1[0])
print("1番目までの要素:",d_array1[:2])
print("最後の要素:",d_array1[-1])
print("0から1番目までの要素:",d_array1[0:2])
print("170以上のインデックス:",np.where(d_array2>170))
print("170以上の要素:",d_array1[d_array2>170])
print("-------\n")

# numpy arrayからdataframeへの変換
# pandas.Seriesを用いて項目名にd_array3の要素（name, height）を設定する
column1 = pd.Series(d_array1,name=d_array3[0])
column2 = pd.Series(d_array2,name=d_array3[1])

# ２つの列をpandas.concatで結合し、dataframeを作成する
d_dataframe = pd.concat([column1,column2],axis=1)

print("d_dataframe:\n",d_dataframe,"\n")
print("name列:\n",d_dataframe["name"],"\n")
print("height列のインデックス0番:",d_dataframe["height"][0],"\n")

column3 = pd.Series(np.array([21,19,30]),name="age")
d_dataframe = pd.concat([column1,column2,column3],axis=1)
print("d_dataframe(add:age):\n",d_dataframe,"\n")
