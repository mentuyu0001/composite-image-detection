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

from model import MyNet     #この後、定義するmodel.pyからのネットワーククラス

import dct_style_learning
import psd_style_learning
import acf_style_learning
import RGBei_style_learning
import RGBlbp_style_learning
import image_style_learning

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

    parser.add_argument("-ep", "--epoch", type=int, default=200)
    args = parser.parse_args()

    # メイン関数
    
    
    try:
        print("Starting DCT Learning")
        dct_style_learning.Main(args)
        print("Finished DCT Learning")
    except Exception as e:
        print(f"DCT!!! ドンマイw\nError: {e}")

    print("\nNext...\n")

    try:
        print("Starting PSD Learning")
        psd_style_learning.Main(args)
        print("Finished PSD Learning")
    except Exception as e:
        print(f"PSD!!! ドンマイw\nError: {e}")

    print("\nNext...\n")

    try:
        print("Starting ACF Learning")
        acf_style_learning.Main(args)
        print("Finished ACF Learning")
    except Exception as e:
        print(f"ACF!!! ドンマイw\nError: {e}")

    print("\nNext...\n")

    try:
        print("Starting RGB_EI Learning")
        RGBei_style_learning.Main(args)
        print("Finished RGB_EI Learning")
    except Exception as e:
        print(f"RGB_EI!!! ドンマイw\nError: {e}")
    
    print("\nNext...\n")

    try:
        print("Starting RGB_LBP Learning")
        RGBlbp_style_learning.Main(args)
        print("Finished RGB_LBP Learning")
    except Exception as e:
        print(f"RGB_LBP!!! ドンマイw\nError: {e}")
    
    print("\nNext...\n")

    try:
        print("Starting IMAGE Learning")
        image_style_learning.Main(args)
        print("Finished IMAGE Learning")
    except Exception as e:
        print(f"IMAGE!!! ドンマイw\nError: {e}")

    
    

