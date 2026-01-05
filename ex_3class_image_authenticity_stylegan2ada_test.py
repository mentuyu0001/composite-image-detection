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
import shutil
import random
from PIL import Image
from copy import deepcopy

from authenticity_model import MyNet     #この後、定義するmodel.pyからのネットワーククラス

# <summary>*****************************************
# 画像特徴抽出関数
# </summary>****************************************

# DCTの計算（離散コサイン変換）、入力形式はnumpy、出力形式はテンソル
def culc_dct(image):

    dct_rows = dct(image, axis=0, norm='ortho')
    dct_result = dct(dct_rows, axis=1, norm='ortho')
    dct_tensor = torch.tensor(dct_result)  # numpyからテンソルへ変換

    return dct_tensor

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

# EIの計算（エッジ情報）、入力形式はnumpy、出力形式はテンソル
def culc_ei_rgbscale(image):

    # image が torch.Tensor 型の場合、numpy 配列に変換
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]

    # すでに numpy.ndarray 型の場合はそのまま使用
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 2:  # グレースケール画像の場合
            image = np.expand_dims(image, axis=-1)  # [H, W] -> [H, W, 1]

    # 画像が uint8 型でない場合、uint8 に変換
    if image.dtype != np.uint8:
        image = np.uint8(image * 255)  # 正規化されている場合、255倍して戻す

    # テンソルからnumpy配列に戻す（[チャネル数, 高さ, 幅] -> [高さ, 幅, チャネル数]）
    #image = image.numpy().transpose(1, 2, 0)  # ここでチャネル軸を最後に移動

    # グレースケールに変換
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # eiの計算
    radius = 1
    n_points = 8 * radius

    
    # 各チャネルにeiを適用
    ei_channels = []
    for c in range(image.shape[2]):  # チャネル数分繰り返す
        channel = image[:, :, c]  # 現在のチャネルを取得
        channel = cv2.GaussianBlur(channel, (5, 5), 0)  # ノイズ除去のための平滑化
        edges = cv2.Canny(channel.astype(np.uint8), 50, 150)  # エッジ検出
        ei_channels.append(edges)

    # 3つのei結果を結合してRGB形式に戻す ([H, W, C])
    ei_result_rgb = np.stack(ei_channels, axis=-1)

    # numpyからテンソルに変換 ([H, W, C] -> [C, H, W])
    ei_tensor = torch.tensor(ei_result_rgb, dtype=torch.float32).permute(2, 0, 1)

    return ei_tensor

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
    dct_features = []
    lbp_features = []
    ei_features = []
    imgs = []
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

        #dct
        # 画像をリサイズ
        dctimg = transform(img).numpy()  # PIL Imageでリサイズした後、numpyに変換
        # 画像特徴をセット
        dct_features.append(culc_dct(dctimg))

        #img
        imgimg = TF.to_tensor(img)
        imgimg = transform(imgimg)
        imgs.append(imgimg)

        #ei
        # BGR -> RGBに変換
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img_rgb)  # 画像をテンソル形式に変換 ([C, H, W])
        # 画像特徴をセット
        ei_features.append(culc_ei_rgbscale(img_tensor))

        #lbp
        lbpimg = transform(img).unsqueeze(0)  # 画像をテンソル形式に変換し、バッチ次元を追加
        # 画像特徴をセット
        lbp_features.append(culc_lbp_rgbscale(lbpimg[0]))

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

        #dct
        # 画像をリサイズ
        dctimg = transform(img).numpy()  # PIL Imageでリサイズした後、numpyに変換
        # 画像特徴をセット
        dct_features.append(culc_dct(dctimg))

        #img
        imgimg = TF.to_tensor(img)
        imgimg = transform(imgimg)
        imgs.append(imgimg)

        #ei
        # BGR -> RGBに変換
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img_rgb)  # 画像をテンソル形式に変換 ([C, H, W])
        # 画像特徴をセット
        ei_features.append(culc_ei_rgbscale(img_tensor))

        #lbp
        lbpimg = transform(img).unsqueeze(0)  # 画像をテンソル形式に変換し、バッチ次元を追加
        # 画像特徴をセット
        lbp_features.append(culc_lbp_rgbscale(lbpimg[0]))

        # 正解ラベルをlabelsにセット
        ans = 1 # 0が実画像、1がPGGAN、2がStarGAN、3がStyleGAN
        labels.append(ans)
    
    #StarGAN画像のセット
    # 画像ファイル名を全て取得
    img_paths = os.path.join(stargan_dir_path, "*.png")
    img_path_list = glob.glob(img_paths)

    # 各画像データ・正解ラベルを格納する
    for img_path in img_path_list:
        # 画像を読み込み
        img = cv2.imread(img_path)

        #dct
        # 画像をリサイズ
        dctimg = transform(img).numpy()  # PIL Imageでリサイズした後、numpyに変換
        # 画像特徴をセット
        dct_features.append(culc_dct(dctimg))

        #img
        imgimg = TF.to_tensor(img)
        imgimg = transform(imgimg)
        imgs.append(imgimg)

        #ei
        # BGR -> RGBに変換
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img_rgb)  # 画像をテンソル形式に変換 ([C, H, W])
        # 画像特徴をセット
        ei_features.append(culc_ei_rgbscale(img_tensor))

        #lbp
        lbpimg = transform(img).unsqueeze(0)  # 画像をテンソル形式に変換し、バッチ次元を追加
        # 画像特徴をセット
        lbp_features.append(culc_lbp_rgbscale(lbpimg[0]))

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

        #dct
        # 画像をリサイズ
        dctimg = transform(img).numpy()  # PIL Imageでリサイズした後、numpyに変換
        # 画像特徴をセット
        dct_features.append(culc_dct(dctimg))

        #img
        imgimg = TF.to_tensor(img)
        imgimg = transform(imgimg)
        imgs.append(imgimg)

        #ei
        # BGR -> RGBに変換
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img_rgb)  # 画像をテンソル形式に変換 ([C, H, W])
        # 画像特徴をセット
        ei_features.append(culc_ei_rgbscale(img_tensor))

        #lbp
        lbpimg = transform(img).unsqueeze(0)  # 画像をテンソル形式に変換し、バッチ次元を追加
        # 画像特徴をセット
        lbp_features.append(culc_lbp_rgbscale(lbpimg[0]))

        # 正解ラベルをlabelsにセット
        ans = 3 # 0が実画像、1がPGGAN、2がStarGAN、3がStyleGAN
        labels.append(ans)

    imgs_array = np.array(imgs)

    # PyTorchで扱うため、各特長をtensor型にする
    dct_features = torch.stack(dct_features)  # リストからテンソルに変換
    ei_features = torch.stack(ei_features)
    lbp_features = torch.stack(lbp_features)
    IMAGE_features = torch.from_numpy(imgs_array)
    labels = torch.tensor(labels)
    
    # 画像データ・正解ラベルのペアをデータにセットする
    dataset = torch.utils.data.TensorDataset(dct_features, ei_features, lbp_features, IMAGE_features, labels)
    
    # データ数を取得
    data_size = len(labels)

    print(f"train data size : {data_size}")

    # データをキャッシュとして保存
    print(f"Saving processed data to cache: {cache_path}")
    torch.save({"dataset": dataset, "data_size": data_size}, cache_path)
    
    # セットしたデータをバッチサイズごとの配列に入れる。
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    return loader, data_size

def test_load_data(path, target_size=(224, 224)):
    print("test data loading")

    # 画像データ・正解ラベル格納用配列
    dct_features = []
    lbp_features = []
    imgs = []
    labels = []

    # 実画像ファイル名を全て取得
    img_paths = os.path.join(path, "*.png")
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
        # ファイル名を取得
        file_name = os.path.basename(img_path)
        #print(f"Processing file: {file_name}")

        # 画像を読み込み
        img = cv2.imread(img_path)

        #dct
        # 画像をリサイズ
        dctimg = transform(img).numpy()  # PIL Imageでリサイズした後、numpyに変換
        # 画像特徴をセット
        dct_features.append(culc_dct(dctimg))

        #img
        imgimg = TF.to_tensor(img)
        imgimg = transform(imgimg)
        imgs.append(imgimg)

        #ei
        # BGR -> RGBに変換
        #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img_tensor = transform(img_rgb)  # 画像をテンソル形式に変換 ([C, H, W])
        # 画像特徴をセット
        #ei_features.append(culc_ei_rgbscale(img_tensor))

        #lbp
        lbpimg = transform(img).unsqueeze(0)  # 画像をテンソル形式に変換し、バッチ次元を追加
        # 画像特徴をセット
        lbp_features.append(culc_lbp_rgbscale(lbpimg[0]))

        # ここは後で修正する
        # 正解ラベルをlabelsにセット
        begin = file_name.find('-model-') + len('-model-')
        tail = file_name.find('.png')
        ans = int(file_name[begin:tail]) # 0が実画像、1がPGGAN、StarGAN、StyleGAN

        if (ans != 0):
            ans = 1

        labels.append(ans)

    imgs_array = np.array(imgs)

    if (len(labels) != 0):

        # PyTorchで扱うため、各特長をtensor型にする
        dct_features = torch.stack(dct_features)  # リストからテンソルに変換
        #ei_features = torch.stack(ei_features)
        lbp_features = torch.stack(lbp_features)
        IMAGE_features = torch.from_numpy(imgs_array)
        #outputimgs_tensor = torch.stack(outputimgs)
        labels = torch.tensor(labels)
        
        # 画像データ・正解ラベルのペアをデータにセットする
        dataset = torch.utils.data.TensorDataset(dct_features, lbp_features, IMAGE_features, labels)
        
        # データ数を取得
        data_size = len(labels)

        print(f"train data size : {data_size}")
        
        # セットしたデータをバッチサイズごとの配列に入れる。
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        
        return loader, data_size
    
    else:
        return False, 0


# <summary>*****************************************
# CNNを用いた学習
# </summary>****************************************

# なし

# <summary>*****************************************
# CNNを用いた推論
# </summary>****************************************

# なし

# <summary>*****************************************
# CNNを用いたテスト
# </summary>****************************************

def authenticityTest(net, device, pgmodel_paths, pgweights, starmodel_paths, starweights, stylemodel_paths, styleweights, 
                            dir_Pg, dir_Star, dir_Style, final_test_corect_counter, final_data_num, final_total_loss, final_f1, kind):
    #初期化達
    realcorrect = 0
    realsize = 0
    realloss = 0
    realf1 = 0

    #モデルのインスタンス化とパラメータの読み込み
    models = []
    for model_path in pgmodel_paths:
        model = deepcopy(net)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)

    # Real画像
    # pg画像の処理
    test_correct_counter = 0    # 正解数カウント
    data_num = 0                # 出力画像用ナンバー

    # f1スコアの初期化
    all_preds = []
    all_labels = []

    # 損失関数の初期化（クロスエントロピー）
    loss_function = torch.nn.CrossEntropyLoss()
    total_loss = 0.0

    loaders, size = test_load_data(dir_Pg)
    
    if (size != 0):

        with torch.no_grad():   # 勾配を計算しない
            for i, (dct_data, lbp_data, imgs, labels) in enumerate(loaders):
                # GPUあるいはCPU用に再構成
                dct_data = dct_data.to(device)      # バッチサイズのデータtensor
                lbp_data = lbp_data.to(device)
                imgs_data = imgs.to(device)
                labels = labels.to(device)  # バッチサイズの正解ラベルtensor

                """
                # 画像を結合する場合はコッチ、現状は加算で対処してる
                # 特徴量を連結
                data = torch.cat((dct_data, psd_data, acf_data), dim=1)  # 特徴量を連結
                """
                
                outputs = []
                for i, (model, weight) in enumerate(zip(models, pgweights)):
                    if i == 0:
                        output = model(dct_data) * weight
                    elif i == 1:
                        output = model(lbp_data) * weight
                    elif i == 2:
                        output = model(imgs_data) * weight
                    outputs.append(output)

                # 重み付き平均を計算
                combined_output = torch.stack(outputs, dim=0).sum(dim=0)

                # クロスエントロピー損失を計算
                loss = loss_function(combined_output, labels)
                total_loss += loss.item()

                # 予測ラベルを取得
                _, predicted = torch.max(combined_output, 1)
                test_correct_counter += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 画像の保存
                data_num += loaders.batch_size


            # F1スコアを計算
            f1 = f1_score(all_labels, all_preds, average="weighted")

            # 平均損失を計算
            average_loss = total_loss / size

            # 加算
            final_data_num += size
            final_test_corect_counter += test_correct_counter
            final_total_loss += total_loss
            final_f1 += f1 * size

            realcorrect += test_correct_counter
            realsize += size
            realloss += total_loss
            realf1 += f1 * size


    # star画像の処理
    test_correct_counter = 0    # 正解数カウント
    data_num = 0                # 出力画像用ナンバー

    # f1スコアの初期化
    all_preds = []
    all_labels = []

    # 損失関数の初期化（クロスエントロピー）
    loss_function = torch.nn.CrossEntropyLoss()
    total_loss = 0.0

    # star画像の処理
    loaders, size = test_load_data(dir_Star)

    if (size != 0):

        #モデルのインスタンス化とパラメータの読み込み
        models = []
        for model_path in starmodel_paths:
            model = deepcopy(net)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            models.append(model)

        with torch.no_grad():   # 勾配を計算しない
            for i, (dct_data, lbp_data, imgs, labels) in enumerate(loaders):
                # GPUあるいはCPU用に再構成
                dct_data = dct_data.to(device)      # バッチサイズのデータtensor
                lbp_data = lbp_data.to(device)
                imgs_data = imgs.to(device)
                labels = labels.to(device)  # バッチサイズの正解ラベルtensor

                """
                # 画像を結合する場合はコッチ、現状は加算で対処してる
                # 特徴量を連結
                data = torch.cat((dct_data, psd_data, acf_data), dim=1)  # 特徴量を連結
                """
                
                outputs = []
                for i, (model, weight) in enumerate(zip(models, starweights)):
                    if i == 0:
                        output = model(dct_data) * weight
                    elif i == 1:
                        output = model(lbp_data) * weight
                    elif i == 2:
                        output = model(imgs_data) * weight
                    outputs.append(output)

                # 重み付き平均を計算
                combined_output = torch.stack(outputs, dim=0).sum(dim=0)

                # クロスエントロピー損失を計算
                loss = loss_function(combined_output, labels)
                total_loss += loss.item()

                # 予測ラベルを取得
                _, predicted = torch.max(combined_output, 1)
                test_correct_counter += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 画像の保存
                data_num += loaders.batch_size


            # F1スコアを計算
            f1 = f1_score(all_labels, all_preds, average="weighted")

            # 平均損失を計算
            average_loss = total_loss / size

            # 加算
            final_data_num += size
            final_test_corect_counter += test_correct_counter
            final_total_loss += total_loss
            final_f1 += f1 * size

            realcorrect += test_correct_counter
            realsize += size
            realloss += total_loss
            realf1 += f1 * size

    # style画像の処理
    test_correct_counter = 0    # 正解数カウント
    data_num = 0                # 出力画像用ナンバー

    # f1スコアの初期化
    all_preds = []
    all_labels = []

    # 損失関数の初期化（クロスエントロピー）
    loss_function = torch.nn.CrossEntropyLoss()
    total_loss = 0.0

    loaders, size = test_load_data(dir_Style)

    if (size != 0):

        #モデルのインスタンス化とパラメータの読み込み
        models = []
        for model_path in stylemodel_paths:
            model = deepcopy(net)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            models.append(model)

        with torch.no_grad():   # 勾配を計算しない
            for i, (dct_data, lbp_data, imgs, labels) in enumerate(loaders):
                # GPUあるいはCPU用に再構成
                dct_data = dct_data.to(device)      # バッチサイズのデータtensor
                lbp_data = lbp_data.to(device)
                imgs_data = imgs.to(device)
                labels = labels.to(device)  # バッチサイズの正解ラベルtensor

                """
                # 画像を結合する場合はコッチ、現状は加算で対処してる
                # 特徴量を連結
                data = torch.cat((dct_data, psd_data, acf_data), dim=1)  # 特徴量を連結
                """
                
                outputs = []
                for i, (model, weight) in enumerate(zip(models, styleweights)):
                    if i == 0:
                        output = model(dct_data) * weight
                    elif i == 1:
                        output = model(lbp_data) * weight
                    elif i == 2:
                        output = model(imgs_data) * weight
                    outputs.append(output)

                # 重み付き平均を計算
                combined_output = torch.stack(outputs, dim=0).sum(dim=0)

                # クロスエントロピー損失を計算
                loss = loss_function(combined_output, labels)
                total_loss += loss.item()

                # 予測ラベルを取得
                _, predicted = torch.max(combined_output, 1)
                test_correct_counter += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 画像の保存
                data_num += loaders.batch_size


            # F1スコアを計算
            f1 = f1_score(all_labels, all_preds, average="weighted")

            # 平均損失を計算
            average_loss = total_loss / size

            # 加算
            final_data_num += size
            final_test_corect_counter += test_correct_counter
            final_total_loss += total_loss
            final_f1 += f1 * size

            realcorrect += test_correct_counter
            realsize += size
            realloss += total_loss
            realf1 += f1 * size

    # 結果保存
    with open("./Results/FinalResults/3class_features_classification.txt", "a") as ftw:
        if (kind == 0):
            ftw.write(f"\nReal Images Score\n")
        elif (kind == 1):
            ftw.write(f"\nPGGAN Images Score\n")
        elif (kind == 2):
            ftw.write(f"\nStarGAN Images Score\n")
        elif (kind == 3):
            ftw.write(f"\nStyleGAN Images Score\n")
        elif (kind == 4):
            ftw.write(f"\nFaceAPP Images Score\n")
        elif (kind == 5):
            ftw.write(f"\nStyleGAN2ADA Images Score\n")

        ftw.write(f"Train correct / total : {str(realcorrect)} / {str(realsize)}\n")
        ftw.write(f"Train Accuracy: {str(realcorrect / realsize)}\n")
        ftw.write(f"Train F1 Score: {str(realf1 / realsize)}\n")
        ftw.write(f"Train Cross-Entropy Loss: {str(realloss / realsize)}\n")

    return final_test_corect_counter, final_data_num, final_total_loss, final_f1

def authenticityTestStar(net, device, pgmodel_paths, pgweights, starmodel_paths, starweights, stylemodel_paths, styleweights, 
                            dir_Pg, dir_Star, dir_Style, final_test_corect_counter, final_data_num, final_total_loss, final_f1, kind):
    #初期化達
    realcorrect = 0
    realsize = 0
    realloss = 0
    realf1 = 0

    # star画像の処理
    test_correct_counter = 0    # 正解数カウント
    data_num = 0                # 出力画像用ナンバー

    # f1スコアの初期化
    all_preds = []
    all_labels = []

    # 損失関数の初期化（クロスエントロピー）
    loss_function = torch.nn.CrossEntropyLoss()
    total_loss = 0.0

    # star画像の処理
    loaders, size = test_load_data(dir_Star)


    if (size != 0):

        #モデルのインスタンス化とパラメータの読み込み
        models = []
        for model_path in starmodel_paths:
            model = deepcopy(net)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            models.append(model)

        with torch.no_grad():   # 勾配を計算しない
            for i, (dct_data, lbp_data, imgs, labels) in enumerate(loaders):
                # GPUあるいはCPU用に再構成
                dct_data = dct_data.to(device)      # バッチサイズのデータtensor
                lbp_data = lbp_data.to(device)
                imgs_data = imgs.to(device)
                labels = labels.to(device)  # バッチサイズの正解ラベルtensor

                print(labels)

                """
                # 画像を結合する場合はコッチ、現状は加算で対処してる
                # 特徴量を連結
                data = torch.cat((dct_data, psd_data, acf_data), dim=1)  # 特徴量を連結
                """
                
                outputs = []
                for i, (model, weight) in enumerate(zip(models, starweights)):
                    if i == 0:
                        output = model(dct_data) * weight
                    elif i == 1:
                        output = model(lbp_data) * weight
                    elif i == 2:
                        output = model(imgs_data) * weight
                    outputs.append(output)

                # 重み付き平均を計算
                combined_output = torch.stack(outputs, dim=0).sum(dim=0)

                print(combined_output.shape)
                os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

                # クロスエントロピー損失を計算
                loss = loss_function(combined_output, labels)
                total_loss += loss.item()

                # 予測ラベルを取得
                _, predicted = torch.max(combined_output, 1)
                test_correct_counter += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 画像の保存
                data_num += loaders.batch_size


            # F1スコアを計算
            f1 = f1_score(all_labels, all_preds, average="weighted")

            # 平均損失を計算
            average_loss = total_loss / size

            # 加算
            final_data_num += size
            final_test_corect_counter += test_correct_counter
            final_total_loss += total_loss
            final_f1 += f1 * size

            realcorrect += test_correct_counter
            realsize += size
            realloss += total_loss
            realf1 += f1 * size


    # 結果保存
    with open("./Results/FinalResults/3class_features_classification.txt", "a") as ftw:
        if (kind == 0):
            ftw.write(f"\nReal Images Score\n")
        elif (kind == 1):
            ftw.write(f"\nPGGAN Images Score\n")
        elif (kind == 2):
            ftw.write(f"\nStarGAN Images Score\n")
        elif (kind == 3):
            ftw.write(f"\nStyleGAN Images Score\n")
        elif (kind == 4):
            ftw.write(f"\nFaceAPP Images Score\n")

        ftw.write(f"Train correct / total : {str(realcorrect)} / {str(realsize)}\n")
        ftw.write(f"Train Accuracy: {str(realcorrect / realsize)}\n")
        ftw.write(f"Train F1 Score: {str(realf1 / realsize)}\n")
        ftw.write(f"Train Cross-Entropy Loss: {str(realloss / realsize)}\n")

    return final_test_corect_counter, final_data_num, final_total_loss, final_f1

def cnn_classification_test(net, device, pgmodel_paths, pgweights, starmodel_paths, starweights, stylemodel_paths, styleweights, 
                            dir_RealToPg, dir_RealToStar, dir_RealToStyle,
                            dir_PgToPg, dir_PgToStar, dir_PgToStyle,
                            dir_StarToPg, dir_StarToStar, dir_StarToStyle,
                            dir_StyleToPg, dir_StyleToStar, dir_StyleToStyle,
                            dir_FaceToPg, dir_FaceToStar, dir_FaceToStyle,
                            dir_Style2ToPg, dir_Style2ToStar, dir_Style2ToStyle,
                            history):
    """
    学習したパラメータでテストを実施する。
    """
    
    

    final_test_corect_counter = 0
    final_data_num = 0
    final_total_loss = 0
    final_f1 = 0


    
    
    # style2画像の処理
    final_test_corect_counter, final_data_num, final_total_loss, final_f1 = authenticityTest(net, device, 
                            pgmodel_paths, pgweights, starmodel_paths, starweights, stylemodel_paths, styleweights, 
                            dir_Style2ToPg, dir_Style2ToStar, dir_Style2ToStyle, final_test_corect_counter, final_data_num, final_total_loss, final_f1, 5)

    return



# <summary>*****************************************
# 分類後のデータをフォルダーに保存する
# </summary>****************************************

incrrect = 0
datanumcounter = 0

def imgs_output(imgs, test_pred, target, counter, batch_size):
    global incrrect, datanumcounter
    """
    分類した画像を結果毎に出力する。
    ファイル名：「データ番号-pre-推論結果-ans-正解ラベル.jpg」
    """
    # フォルダがなければ作る.
    # ある場合は中身を消去して再生成する
    dir_style = "./style"
    dir_star = "./star"
    dir_pg = "./pg"
    
    for i in range(batch_size):
        try:
            datanumcounter += 1
            #print(len(imgs))
            #print(counter + i)
            img = imgs[i].permute(1, 2, 0).byte().numpy()
            pred_num = test_pred[i].item()  # 推論結果
            ans = target[i].item()          # 正解ラベル

            if pred_num != ans:
                incrrect += 1

            # ファイル名設定
            data_num = str(datanumcounter).zfill(7)
            img_name = f"{data_num}-model-{ans}.png"

            if pred_num == 0:
                outputfile = random.randint(1,3)
                if (outputfile == 1):
                    fname = os.path.join(dir_pg, img_name)
                elif (outputfile == 2):
                    fname = os.path.join(dir_star, img_name)
                else:
                    fname = os.path.join(dir_style, img_name)
            elif pred_num == 1:
                fname = os.path.join(dir_pg, img_name)
            elif pred_num == 2:
                fname = os.path.join(dir_star, img_name)
            elif pred_num == 3:
                fname = os.path.join(dir_style, img_name)
            
            # 画像保存
            cv2.imwrite(fname, img)
        
        except IndexError:
            break
        

    return

def last_epoch_NG_output(data, test_pred, target, counter):
    """
    不正解した画像を出力する。
    ファイル名：「データ番号-pre-推論結果-ans-正解ラベル.jpg」
    """
    # フォルダがなければ作る
    dir_path = "./NG_photo_CNN/DCT"
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

# なし

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
    star_test_dir1 = args.testDirStarGAN1
    star_test_dir2 = args.testDirStarGAN2

    # StyleGAN
    style_train_dir = args.trainDirStyleGAN
    style_validation_dir = args.validationDirStyleGAN
    style_test_dir = args.testDirStyleGAN

    # FaceApp
    faceapp_train_dir = args.trainDirFaceApp
    faceapp_validation_dir = args.validationDirFaceApp
    faceapp_test_dir = args.testDirFaceApp

    #data
    dir_RealToPg = args.realtopgdata
    dir_RealToStar = args.realtostardata
    dir_RealToStyle = args.realtostyledata
    dir_PgToPg = args.pgtopgdata
    dir_PgToStar = args.pgtostardata
    dir_PgToStyle = args.pgtostyledata
    dir_StarToPg = args.startopgdata
    dir_StarToStar = args.startostardata
    dir_StarToStyle = args.startostyledata
    dir_StyleToPg = args.styletopgdata
    dir_StyleToStar = args.styletostardata
    dir_StyleToStyle = args.styletostyledata
    dir_FaceToPg = args.facetopgdata
    dir_FaceToStar = args.facetostardata
    dir_FaceToStyle = args.facetostyledata
    dir_Style2ToPg = args.style2topgdata
    dir_Style2ToStar = args.style2tostardata
    dir_Style2ToStyle = args.style2tostyledata
    
    # エポック数（学習回数）
    epoch = args.epoch

    #最後test画像で成績を見る
    print("Loading Test")
    
    #test_real_loaders, test_real_data_size = test_load_realdata("D:/GraduationResearch/Features/test_real_cache.pt", ffhq_test_dir, pggan_test_dir, star_test_dir, style_test_dir, faceapp_test_dir)
    #test_pggan_loaders, test_pggan_data_size = test_load_pggandata("D:/GraduationResearch/Features/test_pg_cache.pt", ffhq_test_dir, pggan_test_dir, star_test_dir, style_test_dir, faceapp_test_dir)
    #test_stargan_loaders, test_stargan_data_size = test_load_stargandata("D:/GraduationResearch/Features/test_star_cache.pt", ffhq_test_dir, pggan_test_dir, star_test_dir, style_test_dir, faceapp_test_dir)
    #test_stylegan_loaders, test_stylegan_data_size = test_load_stylegandata("D:/GraduationResearch/Features/test_style_cache.pt", ffhq_test_dir, pggan_test_dir, star_test_dir, style_test_dir, faceapp_test_dir)
    #test_faceapp_loaders, test_faceapp_data_size = test_load_faceappdata("D:/GraduationResearch/Features/test_faceapp_cache.pt", ffhq_test_dir, pggan_test_dir, star_test_dir, style_test_dir, faceapp_test_dir)

    print("Start Test")

    # pthファイルのパスとモデルの重みの指定
    pgmodel_paths = ["params/DCT_pg_params_cnn.pth", "params/RGB_LBP_pg_params_cnn.pth", "params/IMAGE_pg_params_cnn.pth"]
    pgweights = [0.9836, 0.9476, 0.7974]
    pgweights = [w / sum(pgweights) for w in pgweights]  # 合計が1になるように正規化

    starmodel_paths = ["params/DCT_star_params_cnn.pth", "params/RGB_LBP_star_params_cnn.pth", "params/IMAGE_star_params_cnn.pth"]
    starweights = [0.9999, 1.0, 0.9951]
    starweights = [w / sum(starweights) for w in starweights]  # 合計が1になるように正規化

    stylemodel_paths = ["params/DCT_style_params_cnn.pth", "params/RGB_LBP_style_params_cnn.pth", "params/IMAGE_style_params_cnn.pth"]
    styleweights = [0.9177, 0.9177, 0.5855]
    styleweights = [w / sum(styleweights) for w in styleweights]  # 合計が1になるように正規化

    # ネットワークを構築（ : torch.nn.Module は型アノテーション）
    # 変数netに構築するMyNet()は「ネットワークの実装」で定義します
    # 3つのデータを使うので、その分用意する
    net : torch.nn.Module = MyNet()

    cnn_classification_test(net, device, pgmodel_paths, pgweights, starmodel_paths, starweights, stylemodel_paths, styleweights, 
                            dir_RealToPg, dir_RealToStar, dir_RealToStyle,
                            dir_PgToPg, dir_PgToStar, dir_PgToStyle,
                            dir_StarToPg, dir_StarToStar, dir_StarToStyle,
                            dir_StyleToPg, dir_StyleToStar, dir_StyleToStyle,
                            dir_FaceToPg, dir_FaceToStar, dir_FaceToStyle,
                            dir_FaceToPg, dir_FaceToStar, dir_FaceToStyle,
                              history_test)

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
    parser.add_argument("-tsstar1", "--testDirStarGAN1", type=str, default="M:/GraduationResearch/Images/stargan/test1")
    parser.add_argument("-tsstar2", "--testDirStarGAN2", type=str, default="M:/GraduationResearch/Images/stargan/test2")

    parser.add_argument("-trstyle", "--trainDirStyleGAN", type=str, default="M:/GraduationResearch/Images/stylegan_ffhq/train")
    parser.add_argument("-vstyle", "--validationDirStyleGAN", type=str, default="M:/GraduationResearch/Images/stylegan_ffhq/validation")
    parser.add_argument("-tsstyle", "--testDirStyleGAN", type=str, default="M:/GraduationResearch/Images/stylegan_ffhq/test")

    parser.add_argument("-trreal", "--trainDirFFHQ", type=str, default="M:/GraduationResearch/Images/ffhq/train")
    parser.add_argument("-vreal", "--validationDirFFHQ", type=str, default="M:/GraduationResearch/Images/ffhq/validation")
    parser.add_argument("-tsreal", "--testDirFFHQ", type=str, default="M:/GraduationResearch/Images/ffhq/test")

    parser.add_argument("-trface", "--trainDirFaceApp", type=str, default="M:/GraduationResearch/Images/faceapp/train")
    parser.add_argument("-vface", "--validationDirFaceApp", type=str, default="M:/GraduationResearch/Images/faceapp/validation")
    parser.add_argument("-tsface", "--testDirFaceApp", type=str, default="M:/GraduationResearch/Images/faceapp/test")

    parser.add_argument("-realtopg", "--realtopgdata", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/RealToPg")
    parser.add_argument("-realtostar", "--realtostardata", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/RealToStar")
    parser.add_argument("-realtostyle", "--realtostyledata", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/RealToStyle")
    parser.add_argument("-pgtopg", "--pgtopgdata", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/PgToPg")
    parser.add_argument("-pgtostar", "--pgtostardata", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/PgToStar")
    parser.add_argument("-pgtostyle", "--pgtostyledata", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/PgToStyle")
    parser.add_argument("-startopg", "--startopgdata", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/StarToPg")
    parser.add_argument("-startostar", "--startostardata", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/StarToStar")
    parser.add_argument("-startostyle", "--startostyledata", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/StarToStyle")
    parser.add_argument("-styletopg", "--styletopgdata", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/StyleToPg")
    parser.add_argument("-styletostar", "--styletostardata", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/StyleToStar")
    parser.add_argument("-styletostyle", "--styletostyledata", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/StyleToStyle")
    parser.add_argument("-facetopg", "--facetopgdata", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/FaceToPg")
    parser.add_argument("-facetostar", "--facetostardata", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/FaceToStar")
    parser.add_argument("-facetostyle", "--facetostyledata", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/FaceToStyle")

    parser.add_argument("-style2topg", "--style2topgdata", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/Style2adaToPg")
    parser.add_argument("-style2tostar", "--style2tostardata", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/Style2adaToStar")
    parser.add_argument("-style2tostyle", "--style2tostyledata", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/Style2adaToStyle")

    parser.add_argument("-ep", "--epoch", type=int, default=1)
    args = parser.parse_args()

    # メイン関数
    Main(args)