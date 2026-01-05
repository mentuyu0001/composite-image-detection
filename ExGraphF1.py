import matplotlib.pyplot as plt
import numpy as np

# データの準備
categories = ['One-step + Ensemble', 'Two-step + Ensemble',
               'One-step + Image Only', 'Two-step + Image Only']
values = [0.9703, 0.9308, 0.8826, 0.8042]

# 並び替え
sorted_indices = np.argsort(values)  # 昇順に並べ替えたインデックス
categories = [categories[i] for i in sorted_indices]
values = [values[i] for i in sorted_indices]

# グラフのサイズを指定（横方向に大きく）
plt.figure(figsize=(14, 7))  # 横12インチ、縦6インチ

# 横棒グラフを作成
plt.barh(categories, values, color='lightgreen', height=0.6)

# x軸の範囲を設定（最大値を100に）
plt.xlim(0, 1)

# メモリの間隔を設定（10ずつ）
plt.xticks(np.arange(0, 1.1, 0.1), fontsize=16)
# メモリの間隔を設定（10ずつ）
plt.yticks(fontsize=16)

plt.subplots_adjust(left=0.25)  # 左側の余白を広げる

# 右側と上側の外枠（スパイン）を消す
ax = plt.gca()  # 現在のグラフの軸を取得
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# グラフのタイトルとラベル
plt.xlabel('F1 Score', fontsize=20)

for i, v in enumerate(values):
    plt.text(v + 0.01, i, f'{v:.4f}', color='black', va='center', fontsize=16)


# グリッドの追加
plt.grid(axis='x', linestyle='--', alpha=0.7)

# グラフを表示
plt.show()
