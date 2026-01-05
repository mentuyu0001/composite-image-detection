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

target_size=(224, 224)

# リサイズとテンソル変換のためのtransform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(target_size),  # リサイズ
    transforms.ToTensor()  # テンソルに変換
])

outputimgs = []
imgs = []

img_paths = os.path.join("M:/GraduationResearch/Images/faceapp/test", "*.png")
img_path_list = glob.glob(img_paths)

# 各画像データ・正解ラベルを格納する
for img_path in img_path_list:
    # ファイル名を取得
    file_name = os.path.basename(img_path)
    print(f"Processing file: {file_name}")
    
    # 画像を読み込み
    img = cv2.imread(img_path)

    # output用の保存
    outputimgs.append(img)

    #dct
    # 画像をリサイズ
    #dctimg = transform(img).numpy()  # PIL Imageでリサイズした後、numpyに変換
    # 画像特徴をセット
    #dct_features.append(culc_dct(dctimg))

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
    #lbpimg = transform(img).unsqueeze(0)  # 画像をテンソル形式に変換し、バッチ次元を追加
    # 画像特徴をセット
    #lbp_features.append(culc_lbp_rgbscale(lbpimg[0]))

    # 正解ラベルをlabelsにセット
    #ans = 1 # 0が実画像、1がPGGAN、2がStarGAN、3がStyleGAN
    #labels.append(ans)