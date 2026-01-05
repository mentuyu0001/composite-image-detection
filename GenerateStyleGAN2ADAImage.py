import torch
import dnnlib
import legacy

import os
import numpy as np
import tensorflow as tf
from PIL import Image
from stylegan2_ada import dnnlib
from stylegan2_ada import tflib
from stylegan2_ada import legacy

# モデルファイルのパス
model_path = 'ffhq.pkl'  # 必要に応じてパスを変更

# 出力ディレクトリ
output_dir = './generated_images'
os.makedirs(output_dir, exist_ok=True)

# 画像サイズ
image_size = 299

# 画像生成の数
num_images = 5000

# TensorFlowとStyleGAN2-ADAの設定
tflib.init_tf()

# モデルの読み込み
with open(model_path, 'rb') as f:
    generator_kwargs = dnnlib.EasyDict()
    generator_kwargs.batch_size = 1  # バッチサイズは1に設定
    generator_kwargs.randomize_noise = True

    # モデルの読み込み
    _G, _D, Gs = legacy.load_network_pkl(f)

# 画像生成と保存
for i in range(num_images):
    # ランダムな潜在ベクトルzを生成
    latents = np.random.randn(1, Gs.input_shape[1])

    # 画像を生成
    images = Gs.run(latents, None, **generator_kwargs)

    # 画像を299x299にリサイズ
    img = Image.fromarray(np.clip(images[0], 0, 255).astype(np.uint8))
    img = img.resize((image_size, image_size), Image.LANCZOS)

    # 画像を保存
    img.save(os.path.join(output_dir, f'image_{i+1:04d}.png'))

    if (i+1) % 100 == 0:
        print(f'{i+1} images generated.')

print('Image generation completed.')