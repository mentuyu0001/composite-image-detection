import matplotlib.pyplot as plt
import numpy as np

# データの準備
categories = ['DCT', 'LBP', 'Edge', 'Image', 'ACF', "PSD"]
values = [95.63, 92.17, 75.81, 72.55, 71.53, 52.62]

# 並び替え
sorted_indices = np.argsort(values)  # 昇順に並べ替えたインデックス
categories = [categories[i] for i in sorted_indices]
values = [values[i] for i in sorted_indices]

# グラフのサイズを指定（横方向に大きく）
plt.figure(figsize=(10, 7))  # 横12インチ、縦6インチ

# 横棒グラフを作成
plt.barh(categories, values, color='skyblue', height=0.6)

# x軸の範囲を設定（最大値を100に）
plt.xlim(0, 100)

# メモリの間隔を設定（10ずつ）
plt.xticks(range(0, 101, 10), fontsize=16)
# メモリの間隔を設定（10ずつ）
plt.yticks(fontsize=16)
# 右側と上側の外枠（スパイン）を消す
ax = plt.gca()  # 現在のグラフの軸を取得
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# グラフのタイトルとラベル
plt.xlabel('Accuracy [%]', fontsize=20)

for i, v in enumerate(values):
    plt.text(v + 1, i, str(v), va='center', fontsize=16)


# グリッドの追加
plt.grid(axis='x', linestyle='--', alpha=0.7)

# グラフを表示
plt.show()
