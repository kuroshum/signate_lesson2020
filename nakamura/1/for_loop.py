import numpy as np

values = np.array([10, 3, 1, 5, 8, 6])

#---------------
# 通常のfor文
# 空のarray
passed_values = np.array([])
for ind in np.arange(len(values)):
	#通常のif文
	if values[ind] > 5:
		passed_values = np.append(passed_values,values[ind])

# 結果を標準出力
passed_values = passed_values.astype('int') #int型にキャスト
print("5以上の値",passed_values)
#---------------

#---------------
# リスト内包表記のfor文
passed_values = values[[True if values[ind] > 5 else False for ind in np.arange(len(values))]]

# 結果を標準出力
print("5以上の値",passed_values)
#---------------

convert_values = np.array([])
for j in np.arange(len(values)):
	if values[j] >= 5:
		convert_values = np.append(convert_values,1)
	else:
		convert_values = np.append(convert_values,-1)

convert_values = convert_values.astype('int')
print("values\n",values)
print("convert_values\n",convert_values)
