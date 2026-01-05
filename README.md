# 生成モデルの分類を活用した合成画像の検出精度の向上
(Improved Detection Accuracy of Composite Images Using Classification of Generative Models)

## 概要 (Overview)
本研究は、未知の生成モデルによって作られた偽画像を正確に検出するため、**「モデル分類」** と **「真偽判定」** の2段階のアプローチを用いた検出手法を提案・検証したプロジェクトです。

### 提案手法の構造 (Methodology)
従来の単一の識別器ではなく、生成モデル（AI）の種類を特定してから真偽判定を行う2段階構成（Two-step approach）を採用しました。これにより、各モデルの特徴に特化した精度の高い判定を目指します。
<img width="598" height="341" alt="image" src="https://github.com/user-attachments/assets/02f594c1-3e3c-4813-baa9-f894e7f06e11" />

## 実験結果 (Results)

### 1. 特徴量の有効性検証 (Feature Effectiveness)
DCT, LBP, Edge情報など、異なる画像特徴量を用いて4クラス分類（StarGAN, PGGAN, StyleGAN, 自然画像）を行った結果です。
実験の結果、**DCTとLBPが90%以上の精度**を記録し、偽画像検出において非常に堅牢な特徴量であることが確認されました。

| 精度 (Accuracy) | F1スコア (F1 Score) |
| :---: | :---: |
|<img width="787" height="549" alt="image" src="https://github.com/user-attachments/assets/7bd5ac03-fa4a-45d0-85c1-a04d58f446ca" />|<img width="787" height="549" alt="image" src="https://github.com/user-attachments/assets/25f65814-8939-45ab-8832-9372d908d0e4" />|

### 2. 各生成モデルに対する堅牢性 (Robustness per Model)
自然画像 vs 各生成モデルの2クラス分類における精度です。特にStarGANに対しては非常に高い精度が出ていますが、StyleGANなどは特徴量によって差が出る結果となりました。
本研究では、上位3つの特徴量（DCT, LBP, Image）を用いてアンサンブル学習を行う構成を採用しました。

| 精度 (Accuracy) | F1スコア (F1 Score) |
| :---: | :---: |
|<img width="806" height="538" alt="image" src="https://github.com/user-attachments/assets/e13ae8ae-69cb-4a4a-ae94-a6a17948a3c7" />|<img width="806" height="537" alt="image" src="https://github.com/user-attachments/assets/f6e56c9d-1584-4c68-8ad4-efebabf0107b" />|

## 考察と詳細分析 (Discussion & Analysis)

### 1. 提案手法 vs 既存手法 (Comparison)
提案した「2段階分類（Two-step）」と、全ての偽画像をまとめて学習させる「1段階分類（One-step）」の比較結果です。
総合的なスコアでは1段階分類が高い結果となりましたが、**「自然画像の判定（Recall）」においては、提案手法の方が高い精度（+5.44%）** を記録しました。

* **1段階分類（One-step）:** 未知の偽画像を検出する能力（特異度）が高い。
* **提案手法（Two-step）:** 本物を誤って偽物と判定しない能力（再現率）が高い。

| 比較: 精度 (Accuracy) | 比較: F1スコア (F1 Score) |
| :---: | :---: |
|<img width="838" height="420" alt="image" src="https://github.com/user-attachments/assets/e4f8390f-9c12-4260-9928-336e3de62345" />|<img width="838" height="419" alt="image" src="https://github.com/user-attachments/assets/bc06f7d8-33d8-4d26-9971-eda1b2a5f1f9" />|

#### 自然画像に対する判定精度 (Performance on Natural Images)
以下の表は、自然画像テストデータ（9000枚）に対する分類結果の比較です。提案手法（2-Step）の方が、本物を正しく本物と見抜く能力に優れていることがわかります。

| 項目 (Metric) | 1ステップ分類 + アンサンブル<br>(1-Step + Ensemble) | 2ステップ分類 + アンサンブル<br>(2-Step + Ensemble) |
| :--- | :---: | :---: |
| **正解数 (Correct Count)** | 7780 / 9000 | **8269 / 9000** |
| **精度 (Accuracy)** | 86.44% | **91.88%** |
| **F1スコア (F1 Score)** | 0.9273 | **0.9577** |

### 2. 未知のモデルに対する課題 (Challenge with Unknown Models)
FaceApp（未知のモデル）などのデータに対し、2段階分類のスコアが伸び悩んだ要因の詳細分析です。
1ステップ分類では「偽物」と大雑把に捉えることができる一方、2段階分類では第1段階で特定のモデル（PGGANなど）に無理やり当てはめようとして失敗し、結果として「自然画像」と誤判定してしまうケース（FaceAppで3290枚）が多いことが判明しました。

| 対象データ | 項目 | 1ステップ分類<br>+ アンサンブル | 2ステップ分類<br>+ アンサンブル |
| :--- | :--- | :---: | :---: |
| **FaceApp** | **正解数 (Correct)** | **2938 / 4501** | 943 / 4501 |
| | *判定内訳 (Breakdown)* | - | 自然画像: 3290<br>StarGAN: 0<br>PGGAN v2: 1205<br>StyleGAN: 6 |
| **PGGAN v1** | **正解数 (Correct)** | **8832 / 8970** | 8614 / 8970 |
| | *判定内訳 (Breakdown)* | - | 自然画像: 414<br>StarGAN: 0<br>PGGAN v2: 8556<br>StyleGAN: 0 |

### 3. 改良手法の検討と成果 (Method Improvement)
第1ステップの分類において、自然画像と判定されたものの振り分けを単純なランダムではなく、**「比率分類（Ratio Classification）」** に変更する改良を行いました。
これにより、PGGAN v1（未知のモデル）に対する検出精度が **2.10% 向上** し、提案手法の弱点を補えることが確認されました。

| 項目 (Metric) | 1ステップ分類<br>+ アンサンブル | 2ステップ分類 (旧手法)<br>ランダム分類 | 2ステップ分類 (新手法)<br>**比率分類 (Ratio)** |
| :--- | :---: | :---: | :---: |
| **正解数** | 8832 / 8970 | 8614 / 8970 | **8802 / 8970** |
| **精度 (Accuracy)** | 98.46% | 96.03% | **98.13%** |
| **F1スコア** | 0.9922 | 0.9663 | **0.9905** |

## まとめ (Conclusion)
本研究により、**DCTとLBP**が偽画像検出において極めて有効であることが示されました。
また、提案する2段階モデルは、未知の偽画像に対する検出力（特異度）に課題を残すものの、**「自然画像を正しく自然画像と判定する能力（再現率）」** においては従来手法よりも優れていることが実証されました。今後は、未知のモデルの特徴をより柔軟に捉える第1段階の分類器の改良が課題となります。

## 論文 (Thesis)
* [生成モデルの分類を活用した合成画像の検出精度の向上.pdf](https://github.com/user-attachments/files/24435671/default.pdf)

