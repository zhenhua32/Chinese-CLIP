"""
预先批量编码图片向量, 保存到本地
"""

import os
import json
import io
import base64

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from model import ClipModel


def base64_to_image(image_base64: str) -> Image.Image:
    """
    base64字符串转Image对象
    """
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes))
    return image


def batch_encode_image():
    """
    批量计算图片向量并保存
    """
    image_vectors_file = r"G:\code\github\Chinese-CLIP\datapath\datasets\MUGE\image_vectors.npy"
    index2info_file = r"G:\code\github\Chinese-CLIP\datapath\datasets\MUGE\index2image.json"

    model_path = r"G:\code\github\Chinese-CLIP\datapath\experiments\muge_finetune_vit-b-16_roberta-base_bs160\checkpoints\epoch_latest.pt"
    clip_model = ClipModel(model_path)

    file_list = [
        r"G:\code\github\Chinese-CLIP\datapath\datasets\MUGE\train_imgs.tsv",
        r"G:\code\github\Chinese-CLIP\datapath\datasets\MUGE\valid_imgs.tsv",
    ]
    index2id = dict()
    index = 0
    vector_list = list()
    for file in file_list:
        with open(file, "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines()):
                image_id, image_base64 = line.strip().split("\t")
                index2id[index] = {
                    "image_id": image_id,
                    "image_base64": image_base64,
                }
                image = base64_to_image(image_base64)
                image_vector = clip_model.encode_image(image)
                vector_list.append(image_vector)
                index += 1

    # 保存向量
    vector_list = np.array(vector_list).astype(np.float32)
    print(vector_list.shape, vector_list.dtype)
    np.save(image_vectors_file, vector_list)

    # 保存id映射
    with open(index2info_file, "w", encoding="utf-8") as f:
        json.dump(index2id, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    batch_encode_image()
