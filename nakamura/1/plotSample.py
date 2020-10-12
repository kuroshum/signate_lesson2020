# -*- coding: utf-8 -*-

import matplotlib.pylab as plt
import numpy as np

# 一様分布に従ってランダムサンプル
x1 = np.random.rand(100)

# 正規分布に従ってランダムサンプル
x2 = np.random.randn(100)

# x1とx2のプロット：plot(データ, "マーカー", color="RGBカラーコード")
plt.plot(x1,"o-",color="#FF0000")
plt.plot(x2,"^-",color="#0000FF")

# レジェンド：legend(("データ名1", "データ名2"), fontsize=フォントサイズ)
plt.legend(('uniform random','normal random'),fontsize=14)

# タイトル：title("タイトル名", "データ名2", fontsize=フォントサイズ)
plt.title("random values",fontsize=14)

# 各軸のラベル：xlabel("ラベル名", fontsize=フォントサイズ)
plt.xlabel("sample index",fontsize=14)
plt.ylabel("sample value",fontsize=14)

# ファイルに保存
plt.savefig("plotSample.png")

# グラフの表示
plt.show()
