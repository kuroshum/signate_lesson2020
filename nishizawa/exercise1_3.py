import numpy as np

values = np.array([10, 3, 1, 5, 8, 6])
print(values)

#---------------
# 通常のfor文
# 空のarray
convert_values = []
for ind in np.arange(len(values)):
	#通常のif文
    if values[ind] >= 5:
        convert_values.append(1)
    else :
        convert_values.append(-1)


# 結果を標準出力
print("convert_values: ",convert_values)
#---------------