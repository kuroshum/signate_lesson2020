# coding: utf-8

import pandas as pd
import numpy as np


def exercise1():
	name = np.array(['Yamada Taro', 'Yamada Hanako', 'Wakayama Dai'])
	height = np.array([172.5, 160.5, 180.2])
	age = np.array([21, 19, 30])
	labels = ['name', 'height', 'age']

	d_dataframe = pd.concat([
		pd.Series(name, name=labels[0]),
		pd.Series(height, name=labels[1]),
	], axis=1)
	print(d_dataframe)

	d_dataframe = pd.concat([d_dataframe, pd.Series(age, name=labels[2])], axis=1)
	print(d_dataframe)


if __name__ == '__main__':
	exercise1()
