import numpy as np
import pandas as pd


d_list1 = ['Yamada Taro', 'Yamada Hanako']
d_list2 = [172.5, 160.5]
d_list3 = ['name','height']
d_array1 = np.array(d_list1)
d_array2 = np.array(d_list2)
d_array3 = np.array(d_list3)
d_array1 = np.append(d_array1,'Wakayama Dai')
d_array2 = np.append(d_array2,180.2)
column1 = pd.Series(d_array1,name=d_array3[0])
column2 = pd.Series(d_array2,name=d_array3[1])
d_dataframe = pd.concat([column1,column2],axis=1)


d_array3 = np.append(d_array3,"age")
column3 = pd.Series(np.array([21,19,30]),name=d_array3[2])
d_dataframe = pd.concat([d_dataframe,column3],axis=1)
print("d_dataframe:\n",d_dataframe,"\n")