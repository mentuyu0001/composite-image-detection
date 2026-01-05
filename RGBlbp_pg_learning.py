import os
import argparse
import glob

import cv2
import torch
import torchvision
import torch.nn.functional as f
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from scipy.fftpack import dct
from scipy import signal
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
from skimage.feature import local_binary_pattern

from authenticity_model import MyNet     #この後、定義するauthenticity_model.pyからのネットワーククラス

# <summary>*****************************************
# 画像特徴抽出関数
# </summary>****************************************

# lbpの計算（エッジ情報）、入力形式はnumpy、出力形式はテンソル
def culc_lbp_rgbscale(image):

    # テンソルからnumpy配列に戻す（[チャネル数, 高さ, 幅] -> [高さ, 幅, チャネル数]）
    image = image.numpy().transpose(1, 2, 0)  # ここでチャネル軸を最後に移動

    # グレースケールに変換
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # LBPの計算
    radius = 1
    n_points = 8 * radius

    """
    lbp_result = local_binary_pattern(image, n_points, radius, method='uniform')

    # numpyからテンソルに変換
    lbp_tensor = torch.tensor(lbp_result, dtype=torch.float32)

    lbp_tensor = lbp_tensor.unsqueeze(0).repeat(3, 1, 1)  # 3チャンネルに拡張
    """
    
    # 各チャネルにLBPを適用
    lbp_channels = []
    for c in range(image.shape[2]):  # チャネル数分繰り返す
        channel = image[:, :, c]  # 現在のチャネルを取得
        lbp_result = local_binary_pattern(channel, n_points, radius, method='uniform')
        lbp_channels.append(lbp_result)

    # 3つのLBP結果を結合してRGB形式に戻す ([H, W, C])
    lbp_result_rgb = np.stack(lbp_channels, axis=-1)

    # numpyからテンソルに変換 ([H, W, C] -> [C, H, W])
    lbp_tensor = torch.tensor(lbp_result_rgb, dtype=torch.float32).permute(2, 0, 1)

    return lbp_tensor

# <summary>*****************************************
# データのロード
# </summary>****************************************

# Main()で、画像データがあるディレクトリパスをdir_pathとして受け取る
def train_load_data(cache_path, real_dir_path, pggan_dir_path, stargan_dir_path, stylegan_dir_path, target_size=(224, 224)):
    print("train data loading")

    # キャッシュファイルが存在する場合、ロードして終了
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        cached_data = torch.load(cache_path, weights_only=False)
        dataset, data_size = cached_data["dataset"], cached_data["data_size"]

        # セットしたデータをバッチサイズごとの配列に入れる。
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

        return loader, data_size

    # キャッシュがない場合、新規でデータを処理
    print("No cache found. Processing data...")

    # 画像データ・正解ラベル格納用配列
    rgb_lbp_features = []
    labels = []

    # 実画像ファイル名を全て取得
    img_paths = os.path.join(real_dir_path, "*.png")
    img_path_list = glob.glob(img_paths)

    # リサイズとテンソル変換のためのtransform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),  # リサイズ
        transforms.ToTensor()  # テンソルに変換
    ])

    # 実画像のセット
    # 各画像データ・正解ラベルを格納する
    for img_path in img_path_list:
        # 画像を読み込み
        img = cv2.imread(img_path)

        img = transform(img).unsqueeze(0)  # 画像をテンソル形式に変換し、バッチ次元を追加

        # 画像特徴をセット
        rgb_lbp_features.append(culc_lbp_rgbscale(img[0]))

        # ここは後で修正する
        # 正解ラベルをlabelsにセット
        ans = 0 # 0が実画像、1がPGGAN、2がStarGAN、3がStyleGAN
        labels.append(ans)

    #PGGAN画像のセット
    # 画像ファイル名を全て取得
    img_paths = os.path.join(pggan_dir_path, "*.png")
    img_path_list = glob.glob(img_paths)

    # 各画像データ・正解ラベルを格納する
    for img_path in img_path_list:
        # 画像を読み込み
        img = cv2.imread(img_path)

        img = transform(img).unsqueeze(0)  # 画像をテンソル形式に変換し、バッチ次元を追加

        # 画像特徴をセット
        rgb_lbp_features.append(culc_lbp_rgbscale(img[0]))

        # 正解ラベルをlabelsにセット
        ans = 1 # 0が実画像、1がPGGAN、2がStarGAN、3がStyleGAN
        labels.append(ans)
    """
    #StarGAN画像のセット
    # 画像ファイル名を全て取得
    img_paths = os.path.join(stargan_dir_path, "*.png")
    img_path_list = glob.glob(img_paths)

    # 各画像データ・正解ラベルを格納する
    for img_path in img_path_list:
        # 画像を読み込み
        img = cv2.imread(img_path)

        img = transform(img).unsqueeze(0)  # 画像をテンソル形式に変換し、バッチ次元を追加

        # 画像特徴をセット
        rgb_lbp_features.append(culc_lbp_rgbscale(img[0]))

        # 正解ラベルをlabelsにセット
        ans = 2 # 0が実画像、1がPGGAN、2がStarGAN、3がStyleGAN
        labels.append(ans)

    #StyleGAN画像のセット
    # 画像ファイル名を全て取得
    img_paths = os.path.join(stylegan_dir_path, "*.png")
    img_path_list = glob.glob(img_paths)

    # 各画像データ・正解ラベルを格納する
    for img_path in img_path_list:
        # 画像を読み込み
        img = cv2.imread(img_path)

        img = transform(img).unsqueeze(0)  # 画像をテンソル形式に変換し、バッチ次元を追加

        # 画像特徴をセット
        rgb_lbp_features.append(culc_lbp_rgbscale(img[0]))

        # 正解ラベルをlabelsにセット
        ans = 3 # 0が実画像、1がPGGAN、2がStarGAN、3がStyleGAN
        labels.append(ans)
    """
    # PyTorchで扱うため、各特長をtensor型にする
    rgb_lbp_features = torch.stack(rgb_lbp_features)  # リストからテンソルに変換
    labels = torch.tensor(labels)
    
    # 画像データ・正解ラベルのペアをデータにセットする
    dataset = torch.utils.data.TensorDataset(rgb_lbp_features, labels)
    
    # データ数を取得
    data_size = len(labels)

    print(f"train data size : {data_size}")

    # データをキャッシュとして保存
    print(f"Saving processed data to cache: {cache_path}")
    torch.save({"dataset": dataset, "data_size": data_size}, cache_path)
    
    # セットしたデータをバッチサイズごとの配列に入れる。
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    return loader, data_size

def test_load_data(cache_path, real_dir_path, pggan_dir_path, stargan_dir_path, stylegan_dir_path, target_size=(224, 224)):
    print("test data loading")

    """
    キャッシュを利用して、画像データのロード時間を短縮する関数。
    - cache_path: キャッシュファイルの保存先
    - real_dir_path, pggan_dir_path, stargan_dir_path, stylegan_dir_path: 各画像ディレクトリパス
    """
    # キャッシュファイルが存在する場合、ロードして終了
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        cached_data = torch.load(cache_path, weights_only=False)
        dataset, data_size = cached_data["dataset"], cached_data["data_size"]

        # セットしたデータをバッチサイズごとの配列に入れる。
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

        return loader, data_size

    # 画像データ・正解ラベル格納用配列
    rgb_lbp_features = []
    labels = []

    # 実画像ファイル名を全て取得
    img_paths = os.path.join(real_dir_path, "*.png")
    img_path_list = glob.glob(img_paths)

    # リサイズとテンソル変換のためのtransform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),  # リサイズ
        transforms.ToTensor()  # テンソルに変換
    ])

    # 実画像のセット
    # 各画像データ・正解ラベルを格納する
    for img_path in img_path_list:
        # 画像を読み込み
        img = cv2.imread(img_path)

        img = transform(img).unsqueeze(0)  # 画像をテンソル形式に変換し、バッチ次元を追加

        # 画像特徴をセット
        rgb_lbp_features.append(culc_lbp_rgbscale(img[0]))

        # ここは後で修正する
        # 正解ラベルをlabelsにセット
        ans = 0 # 0が実画像、1がPGGAN、2がStarGAN、3がStyleGAN
        labels.append(ans)

    #PGGAN画像のセット
    # 画像ファイル名を全て取得
    img_paths = os.path.join(pggan_dir_path, "*.png")
    img_path_list = glob.glob(img_paths)

    # 各画像データ・正解ラベルを格納する
    for img_path in img_path_list:
        # 画像を読み込み
        img = cv2.imread(img_path)

        img = transform(img).unsqueeze(0)  # 画像をテンソル形式に変換し、バッチ次元を追加

        # 画像特徴をセット
        rgb_lbp_features.append(culc_lbp_rgbscale(img[0]))

        # 正解ラベルをlabelsにセット
        ans = 1 # 0が実画像、1がPGGAN、2がStarGAN、3がStyleGAN
        labels.append(ans)
    """
    #StarGAN画像のセット
    # 画像ファイル名を全て取得
    img_paths = os.path.join(stargan_dir_path, "*.png")
    img_path_list = glob.glob(img_paths)

    # 各画像データ・正解ラベルを格納する
    for img_path in img_path_list:
        # 画像を読み込み
        img = cv2.imread(img_path)

        img = transform(img).unsqueeze(0)  # 画像をテンソル形式に変換し、バッチ次元を追加

        # 画像特徴をセット
        rgb_lbp_features.append(culc_lbp_rgbscale(img[0]))

        # 正解ラベルをlabelsにセット
        ans = 2 # 0が実画像、1がPGGAN、2がStarGAN、3がStyleGAN
        labels.append(ans)

    #StyleGAN画像のセット
    # 画像ファイル名を全て取得
    img_paths = os.path.join(stylegan_dir_path, "*.png")
    img_path_list = glob.glob(img_paths)

    # 各画像データ・正解ラベルを格納する
    for img_path in img_path_list:
        # 画像を読み込み
        img = cv2.imread(img_path)

        img = transform(img).unsqueeze(0)  # 画像をテンソル形式に変換し、バッチ次元を追加

        # 画像特徴をセット
        rgb_lbp_features.append(culc_lbp_rgbscale(img[0]))

        # 正解ラベルをlabelsにセット
        ans = 3 # 0が実画像、1がPGGAN、2がStarGAN、3がStyleGAN
        labels.append(ans)
    """

    # PyTorchで扱うため、各特長をtensor型にする
    rgb_lbp_features = torch.stack(rgb_lbp_features)  # リストからテンソルに変換
    labels = torch.tensor(labels)
    
    # 画像データ・正解ラベルのペアをデータにセットする
    dataset = torch.utils.data.TensorDataset(rgb_lbp_features, labels)
    
    # データ数を取得
    data_size = len(labels)

    print(f"train data size : {data_size}")

    # データをキャッシュとして保存
    print(f"Saving processed data to cache: {cache_path}")
    torch.save({"dataset": dataset, "data_size": data_size}, cache_path)
    
    # セットしたデータをバッチサイズごとの配列に入れる。
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    return loader, data_size

# <summary>*****************************************
# CNNを用いた学習
# </summary>****************************************

def cnn_train(net, device, loaders, data_size, optimizer, e, history):
    """CNNによる学習を実行する。
    net.parameters()に各conv, fcのウェイト・バイアスが格納される。
    """

    # f1スコアの初期化
    all_preds = []
    all_labels = []

    loss_sum = 0        # 損失関数の値（エポック合計）
    train_correct_counter = 0   # 正解数カウント

    # 学習開始（再開）
    net.train(True) # 引数は省略可能

    criterion = torch.nn.CrossEntropyLoss()

    for i, (rgb_lbp_data, labels) in tqdm(enumerate(loaders)):
        # GPUあるいはCPU用に再構成
        rgb_lbp_data = rgb_lbp_data.to(device)      # バッチサイズのデータtensor
        labels = labels.to(device)  # バッチサイズの正解ラベルtensor

        # 特徴量を結合
        data = rgb_lbp_data

        # 学習
        optimizer.zero_grad()   # 前回までの誤差逆伝播の勾配をリセット
        output = net(data)      # 推論を実施（順伝播による出力）
        probs = f.softmax(output, dim=1)

        loss = criterion(output, labels)   # 交差エントロピーによる損失計算（バッチ平均値）
        loss_sum += loss.item() * labels.size(0) # バッチ合計値に直して加算

        loss.backward()         # 誤差逆伝播
        optimizer.step()        # パラメータ更新

        # 予測ラベルを取得
        train_pred = probs.argmax(dim=1, keepdim=False)  # 最も確率が高いクラスを取得
        train_correct_counter += (train_pred == labels).sum().item()  # 正解数を加算

        # F1スコア用にデータ収集
        all_preds.extend(train_pred.cpu().numpy())  # 予測ラベルを保存
        all_labels.extend(labels.cpu().numpy())    # 正解ラベルを保存

        # 進捗を出力（8バッチ分ごと）
        if i % 8 == 0:
            print('Training log: epoch_{} ({} / {}). Loss: {}'.format(e+1, (i+1)*loaders.batch_size, data_size, loss.item()))

    # エポック全体の平均の損失関数、正解率を格納
    ave_loss = loss_sum / data_size
    ave_accuracy = train_correct_counter / data_size
    # F1スコアを計算
    f1 = f1_score(all_labels, all_preds, average="weighted")  # 加重平均を使用

    history['train_loss'].append(ave_loss)
    history['train_acc'].append(ave_accuracy)
    history['train_f1'].append(f1)
    print(f"Train correct / total : {train_correct_counter} / {data_size}")
    print(f"Train Accuracy: {ave_accuracy}")
    print(f"Train F1 Score: {f1}")
    print(f"Train Cross-Entropy Loss: {ave_loss}\n") 
    
    # フォルダがなければ作る
    os.makedirs("./Results/RGB_LBP", exist_ok=True)

    # 結果保存
    with open("./Results/RGB_LBP/RGB_LBP_pg_train.txt", "w") as ftw:
        ftw.write(f"Train correct / total : {str(train_correct_counter)} / {str(data_size)}\n")
        ftw.write(f"Train Accuracy: {str(ave_accuracy)}\n")
        ftw.write(f"Train F1 Score: {str(f1)}\n")
        ftw.write(f"Train Cross-Entropy Loss: {str(ave_loss)}\n")

    return

# <summary>*****************************************
# CNNを用いた推論
# </summary>****************************************

def cnn_validation(net, device, loaders, data_size, e, epoch, history):
    """
    学習したパラメータでテストを実施する。
    """
    # 学習のストップ
    net.eval() # または　net.train(False)でもいい

    # f1スコアの初期化
    all_preds = []
    all_labels = []
    
    loss_sum = 0                # 損失関数の値（数値のみ）
    test_correct_counter = 0    # 正解数カウント
    data_num = 0                # 最終エポックでの出力画像用ナンバー

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, (rgb_lbp_data, labels) in enumerate(loaders):
            # GPUあるいはCPU用に再構成
            rgb_lbp_data = rgb_lbp_data.to(device)      # バッチサイズのデータtensor
            labels = labels.to(device)  # バッチサイズの正解ラベルtensor

            # 特徴量を結合
            data = rgb_lbp_data

            output = net(data)      # 推論を実施（順伝播による出力）

            loss = criterion(output, labels)   # 交差エントロピーによる損失計算（バッチ平均値）
            loss_sum += loss.item() * labels.size(0) # バッチ合計値に直して加算
            
            # 予測ラベルを取得
            test_pred = output.argmax(dim=1, keepdim=False)  # 最も確率が高いクラスを取得
            test_correct_counter += (test_pred == labels).sum().item()  # 正解数を加算

            # F1スコア用にデータ収集
            all_preds.extend(test_pred.cpu().numpy())  # 予測ラベルを保存
            all_labels.extend(labels.cpu().numpy())   # 正解ラベルを保存

            # 最終エポックのみNG画像を出力
            if e == epoch - 1:
                last_epoch_NG_output(data, test_pred, labels, data_num)
                data_num += loaders.batch_size
    
    # F1スコアを計算
    f1 = f1_score(all_labels, all_preds, average="weighted")  # 加重平均を使用

    # テスト全体の平均の損失関数、正解率を格納
    ave_loss = loss_sum / data_size
    ave_accuracy = test_correct_counter / data_size
    history['test_loss'].append(ave_loss)
    history['test_acc'].append(ave_accuracy)
    history['test_f1'].append(f1)
    print(f"Test correct / total : {test_correct_counter} / {data_size}")
    print(f"Test Accuracy: {ave_accuracy}")
    print(f"Test F1 Score: {f1}")
    print(f"Test Cross-Entropy Loss: {ave_loss}\n") 

    # フォルダがなければ作る
    os.makedirs("./Results/RGB_LBP", exist_ok=True)

    # 結果保存
    with open("./Results/RGB_LBP/RGB_LBP_pg_validation.txt", "w") as ftw:
        ftw.write(f"Validation correct / total : {str(test_correct_counter)} / {str(data_size)}\n")
        ftw.write(f"Validation Accuracy: {str(ave_accuracy)}\n")
        ftw.write(f"Validation F1 Score: {str(f1)}\n")
        ftw.write(f"Validation Cross-Entropy Loss: {str(ave_loss)}\n")

    return

def last_epoch_NG_output(data, test_pred, target, counter):
    """
    不正解した画像を出力する。
    ファイル名：「データ番号-pre-推論結果-ans-正解ラベル.jpg」
    """
    # フォルダがなければ作る
    dir_path = "./rgb_LBP_pg__NG_photo_CNN"
    os.makedirs(dir_path, exist_ok=True)

    for i, img in enumerate(data):
        pred_num = test_pred[i].item()  # 推論結果
        ans = target[i].item()          # 正解ラベル

        # 推論結果と正解ラベルを比較して不正解なら画像保存
        if pred_num != ans:
            # ファイル名設定
            data_num = str(counter+i).zfill(5)
            img_name = f"{data_num}-pre-{pred_num}-ans-{ans}.jpg"
            fname = os.path.join(dir_path, img_name)
            
            # 画像保存
            torchvision.utils.save_image(img, fname)

    return

# <summary>*****************************************
# CNNを用いたテスト
# </summary>****************************************

def cnn_test(net, device, model_path, loaders, data_size):
    """
    学習したパラメータでテストを実施する。
    """
    # f1スコアの初期化
    all_preds = []
    all_labels = []

    # 損失関数の初期化（クロスエントロピー）
    loss_function = torch.nn.CrossEntropyLoss()
    total_loss = 0.0

    #モデルのインスタンス化とパラメータの読み込み
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    net.to(device)
    net.eval()

    test_correct_counter = 0    # 正解数カウント

    with torch.no_grad():   # 勾配を計算しない
        for i, (rgb_lbp_data, labels) in enumerate(loaders):
            # GPUあるいはCPU用に再構成
            rgb_lbp_data = rgb_lbp_data.to(device)      # バッチサイズのデータtensor
            labels = labels.to(device)  # バッチサイズの正解ラベルtensor
            
            output = net(rgb_lbp_data)

            loss = loss_function(output, labels)
            total_loss += loss.item()

            # 予測ラベルを取得
            _, predicted = torch.max(output, 1)
            test_correct_counter += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 精度の出力
    accuracy = test_correct_counter / data_size

    # F1スコアを計算
    f1 = f1_score(all_labels, all_preds, average="weighted")

    # 平均損失を計算
    average_loss = total_loss / len(loaders)

    # 結果を出力
    print(f"correct / total: {test_correct_counter} / {data_size}")
    print(f"Test Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Average Cross-Entropy Loss: {average_loss}")

    # フォルダがなければ作る
    os.makedirs("./Results/RGB_LBP", exist_ok=True)

    # 結果保存
    with open("./Results/RGB_LBP/RGB_LBP_pg_test.txt", "w") as ftw:
        ftw.write(f"Test correct / total : {str(test_correct_counter)} / {str(data_size)}\n")
        ftw.write(f"Test Accuracy: {str(accuracy)}\n")
        ftw.write(f"Test F1 Score: {str(f1)}\n")
        ftw.write(f"Test Cross-Entropy Loss: {str(average_loss)}\n")

    return

def last_epoch_NG_output(data, test_pred, target, counter):
    """
    不正解した画像を出力する。
    ファイル名：「データ番号-pre-推論結果-ans-正解ラベル.jpg」
    """
    # フォルダがなければ作る
    dir_path = "./NG_photo_CNN/RGB_LBP_pg_"
    os.makedirs(dir_path, exist_ok=True)

    for i, img in enumerate(data):
        pred_num = test_pred[i].item()  # 推論結果
        ans = target[i].item()          # 正解ラベル

        # 推論結果と正解ラベルを比較して不正解なら画像保存
        if pred_num != ans:
            # ファイル名設定
            data_num = str(counter+i).zfill(5)
            img_name = f"{data_num}-pre-{pred_num}-ans-{ans}.jpg"
            fname = os.path.join(dir_path, img_name)
            
            # 画像保存
            torchvision.utils.save_image(img, fname)

    return

# <summary>*****************************************
# 出力結果の表示
# </summary>****************************************

def output_graph(epoch, history_train, history_test):
    os.makedirs("./CNNLearningResult/RGB_LBP_pg_", exist_ok=True)

    # 各エポックの損失関数グラフ
    plt.figure()
    plt.plot(range(1, epoch+1), history_train['train_loss'], label='train_loss', marker='.')
    plt.plot(range(1, epoch+1), history_test['test_loss'], label='test_loss', marker='.')
    plt.xlabel('epoch')
    plt.legend() # 凡例
    plt.savefig('./CNNLearningResult/RGB_LBP_pg_/loss_cnn.png')

    # 各エポックの正解率グラフ
    plt.figure()
    plt.plot(range(1, epoch+1), history_train['train_acc'], label='train_acc', marker='.')
    plt.plot(range(1, epoch+1), history_test['test_acc'], label='test_acc', marker='.')
    plt.xlabel('epoch')
    plt.legend() # 凡例
    plt.savefig('./CNNLearningResult/RGB_LBP_pg_/acc_cnn.png')

    # 各エポックの正解率グラフ
    plt.figure()
    plt.plot(range(1, epoch+1), history_train['train_f1'], label='train_f1', marker='.')
    plt.plot(range(1, epoch+1), history_test['test_f1'], label='test_f1', marker='.')
    plt.xlabel('epoch')
    plt.legend() # 凡例
    plt.savefig('./CNNLearningResult/RGB_LBP_pg_/f1_cnn.png')

    return

# <summary>*****************************************
# 実行処理
# </summary>****************************************

def Main(args):
    # 計算環境が、CUDA(GPU)か、CPUか
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device: ' + device)

    # 学習・テスト結果の保存用辞書
    history_train = {
        'train_loss': [],   # 損失関数の値
        'train_acc': [],    # 正解率
        'train_f1' : [],    # f1スコア
    }

    history_test = {
        'test_loss': [],    # 損失関数の値
        'test_acc': [],     # 正解率
        'test_f1' : [],    # f1スコア
    }

    # ネットワークを構築（ : torch.nn.Module は型アノテーション）
    # 変数netに構築するMyNet()は「ネットワークの実装」で定義します
    net : torch.nn.Module = MyNet()
    net = net.to(device) # GPUあるいはCPUに合わせて再構成

    # データローダー・データ数を取得
    #（load_dataは「学習データ・テストデータのロードの実装（データローダー）」の章で定義します）
    # 自然画像
    ffhq_train_dir = args.trainDirFFHQ
    ffhq_validation_dir = args.validationDirFFHQ
    ffhq_test_dir = args.testDirFFHQ

    # PGGAN
    pggan_train_dir = args.trainDirPGGAN
    pggan_validation_dir = args.validationDirPGGAN
    pggan_test_dir = args.testDirPGGAN

    # StarGAN
    star_train_dir = args.trainDirStarGAN
    star_validation_dir = args.validationDirStarGAN
    star_test_dir = args.testDirStarGAN

    # StyleGAN
    style_train_dir = args.trainDirStyleGAN
    style_validation_dir = args.validationDirStyleGAN
    style_test_dir = args.testDirStyleGAN

    train_loaders, train_data_size = train_load_data("D:/GraduationResearch/Features/RGB_LBP_pg_train_cache.pt", ffhq_train_dir, pggan_train_dir, star_train_dir, style_train_dir)
    validation_loaders, validation_data_size = test_load_data("D:/GraduationResearch/Features/RGB_LBP_pg_validation_cache.pt", ffhq_validation_dir, pggan_validation_dir, star_validation_dir, style_validation_dir)
    
    # オプティマイザを設定
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.00002)
    
    # エポック数（学習回数）
    epoch = args.epoch

    # 学習・テストを実行
    print("Start Learning")
    for e in range(epoch):
        # 以下2つの関数は「学習の実装」「テストの実装」の章で定義します
        cnn_train(net, device, train_loaders, train_data_size, optimizer, e, history_train)
        cnn_validation(net, device, validation_loaders, validation_data_size, e, epoch, history_test)

    # 学習済みパラメータを保存
    torch.save(net.state_dict(), './params/RGB_LBP_pg_params_cnn.pth')
    print("Finish Learning")

    # 結果を出力（「結果出力の実装」の章で定義します）
    output_graph(epoch, history_train, history_test)

    #最後test画像で成績を見る
    print("Loading Test")
    
    test_loaders, test_data_size = test_load_data("D:/GraduationResearch/Features/RGB_LBP_pg_test_cache.pt", ffhq_test_dir, pggan_test_dir, star_test_dir, style_test_dir)

    print("Start Test")

    # ネットワークを構築（ : torch.nn.Module は型アノテーション）
    # 変数netに構築するMyNet()は「ネットワークの実装」で定義します
    testnet : torch.nn.Module = MyNet()
    testnet = net.to(device) # GPUあるいはCPUに合わせて再構成

    cnn_test(testnet, device, "./params/RGB_LBP_pg_params_cnn.pth", test_loaders, test_data_size)

    print("Finish Test")

if __name__ == '__main__':
    """
    学習を行うプログラム。

    trainDir : 学習用画像があるディレクトリパス
    testDir  : テスト用画像があるディレクトリパス
    epoch    : エポック数
    """

    # 起動引数設定
    parser = argparse.ArgumentParser()
    parser.add_argument("-trpg", "--trainDirPGGAN", type=str, default="M:/GraduationResearch/Images/pggan_v2/train")
    parser.add_argument("-vpg", "--validationDirPGGAN", type=str, default="M:/GraduationResearch/Images/pggan_v2/validation")
    parser.add_argument("-tspg", "--testDirPGGAN", type=str, default="M:/GraduationResearch/Images/pggan_v2/test")

    parser.add_argument("-trstar", "--trainDirStarGAN", type=str, default="M:/GraduationResearch/Images/stargan/train")
    parser.add_argument("-vstar", "--validationDirStarGAN", type=str, default="M:/GraduationResearch/Images/stargan/validation")
    parser.add_argument("-tsstar", "--testDirStarGAN", type=str, default="M:/GraduationResearch/Images/stargan/test")

    parser.add_argument("-trstyle", "--trainDirStyleGAN", type=str, default="M:/GraduationResearch/Images/stylegan_ffhq/train")
    parser.add_argument("-vstyle", "--validationDirStyleGAN", type=str, default="M:/GraduationResearch/Images/stylegan_ffhq/validation")
    parser.add_argument("-tsstyle", "--testDirStyleGAN", type=str, default="M:/GraduationResearch/Images/stylegan_ffhq/test")

    parser.add_argument("-trreal", "--trainDirFFHQ", type=str, default="M:/GraduationResearch/Images/ffhq/train")
    parser.add_argument("-vreal", "--validationDirFFHQ", type=str, default="M:/GraduationResearch/Images/ffhq/validation")
    parser.add_argument("-tsreal", "--testDirFFHQ", type=str, default="M:/GraduationResearch/Images/ffhq/test")

    parser.add_argument("-ep", "--epoch", type=int, default=100)
    args = parser.parse_args()

    # メイン関数
    Main(args)