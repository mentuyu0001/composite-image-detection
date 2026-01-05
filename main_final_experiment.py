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

#from model import MyNet     #この後、定義するmodel.pyからのネットワーククラス

import ex_2class_features_authenticity
import ex_2class_image_authenticity
import ex_3class_features_authenticity
import ex_3class_features_classification
import ex_3class_image_authenticity
import ex_3class_image_classification

import ex_2class_features_authenticity_stylegan2ada
import ex_2class_image_authenticity_stylegan2ada
import ex_3class_features_authenticity_stylegan2ada
import ex_3class_features_classification_stylegan2ada
import ex_3class_image_authenticity_stylegan2ada
import ex_3class_image_classification_stylegan2ada

import ex_3class_image_authenticity_stylegan2ada_test
import ex_3class_image_classification_stylegan2ada_test

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

    parser.add_argument("-tsstyle2", "--testDirstylegan2ada", type=str, default="M:/GraduationResearch/Images/pggan_v1/test")

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

    parser.add_argument("-realtopgimg", "--realtopgdataimg", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/ImageRealToPg")
    parser.add_argument("-realtostarimg", "--realtostardataimg", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/ImageRealToStar")
    parser.add_argument("-realtostyleimg", "--realtostyledataimg", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/ImageRealToStyle")
    parser.add_argument("-pgtopgimg", "--pgtopgdataimg", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/ImagePgToPg")
    parser.add_argument("-pgtostarimg", "--pgtostardataimg", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/ImagePgToStar")
    parser.add_argument("-pgtostyleimg", "--pgtostyledataimg", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/ImagePgToStyle")
    parser.add_argument("-startopgimg", "--startopgdataimg", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/ImageStarToPg")
    parser.add_argument("-startostarimg", "--startostardataimg", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/ImageStarToStar")
    parser.add_argument("-startostyleimg", "--startostyledataimg", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/ImageStarToStyle")
    parser.add_argument("-styletopgimg", "--styletopgdataimg", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/ImageStyleToPg")
    parser.add_argument("-styletostarimg", "--styletostardataimg", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/ImageStyleToStar")
    parser.add_argument("-styletostyleimg", "--styletostyledataimg", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/ImageStyleToStyle")
    parser.add_argument("-facetopgimg", "--facetopgdataimg", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/ImageFaceToPg")
    parser.add_argument("-facetostarimg", "--facetostardataimg", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/ImageFaceToStar")
    parser.add_argument("-facetostyleimg", "--facetostyledataimg", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/ImageFaceToStyle")

    parser.add_argument("-style2topgimg", "--style2topgdataimg", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/ImageStyle2adaToPg")
    parser.add_argument("-style2tostarimg", "--style2tostardataimg", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/ImageStyle2adaToStar")
    parser.add_argument("-style2tostyleimg", "--style2tostyledataimg", type=str, default="M:/GraduationResearch/Programs/LearningFeatures/ImageStyle2adaToStyle")

    parser.add_argument("-ep", "--epoch", type=int, default=1)
    args = parser.parse_args()

    # メイン関数
    
    
    try:
        print("Starting 3class features test")
        #ex_3class_features_classification.Main(args)
        #ex_3class_features_authenticity.Main(args)
        #ex_3class_features_classification_stylegan2ada.Main(args)
        ex_3class_features_authenticity_stylegan2ada.Main(args)
        #ex_3class_image_classification_stylegan2ada_test.Main(args)
        #ex_3class_image_authenticity_stylegan2ada_test.Main(args)
        print("Finished 3class features test")
    except Exception as e:
        print(f"3class features test Error: {e}")

    print("\nNext...\n")
    
    try:
        print("Starting 3class image test")
        #ex_3class_image_classification.Main(args)
        #ex_3class_image_authenticity.Main(args)
        #ex_3class_image_classification_stylegan2ada.Main(args)
        #ex_3class_image_authenticity_stylegan2ada.Main(args)
        print("Finished 3class image test")
    except Exception as e:
        print(f"3class image test Error: {e}")

    print("\nNext...\n")

    try:
        print("Starting 2class features test")
        #ex_2class_features_authenticity.Main(args)
        #ex_2class_features_authenticity_stylegan2ada.Main(args)
        print("Finished 2class features test")
    except Exception as e:
        print(f"2class features test Error: {e}")

    print("\nNext...\n")

    try:
        print("Starting 2class image test")
        #ex_2class_image_authenticity.Main(args)
        #ex_2class_image_authenticity_stylegan2ada.Main(args)
        print("Finished 2class image test")
    except Exception as e:
        print(f"2class image test Error: {e}")