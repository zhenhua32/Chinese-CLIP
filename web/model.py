"""
加载模型, 并提供文本和图片编码功能
"""

import os
import json
import io
import base64
from typing import Union

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

import cn_clip.clip as clip
from cn_clip.clip.model import CLIP
from cn_clip.clip import load_from_name


def base64_to_image(image_base64: str) -> Image.Image:
    """
    base64字符串转Image对象
    """
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes))
    return image


class ClipModel:
    def __init__(self, model_path: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.image_preprocess = self.load_model(model_path)
        self.model.eval()

    def load_model(self, model_path: str):
        """
        加载模型和图片处理函数
        """
        model, image_preprocess = load_from_name(
            model_path,
            device=self.device,
            vision_model_name="ViT-B-16",
            text_model_name="RoBERTa-wwm-ext-base-chinese",
            input_resolution=224,
            convert_to_float16=True,
        )
        model: CLIP = model.eval()
        return model, image_preprocess

    def encode_text(self, text: str) -> np.ndarray:
        """
        计算文本向量
        """
        text = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0]

    def encode_image(self, image_or_file: Union[Image.Image, str]) -> np.ndarray:
        """
        计算图片向量
        """
        if isinstance(image_or_file, Image.Image):
            return self.encode_image_by_image(image_or_file)
        elif isinstance(image_or_file, str):
            return self.encode_image_by_file(image_or_file)
        else:
            raise TypeError("image_or_file must be Image or str")

    def encode_image_by_image(self, image: Image) -> np.ndarray:
        """
        计算图片向量
        """
        image = self.image_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0]

    def encode_image_by_file(self, image_path: str) -> np.ndarray:
        """
        计算图片向量
        """
        image = self.image_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0]

    def batch_encode_image(self, image_or_files: list, batch_size: int = 128) -> np.ndarray:
        """
        计算图片向量
        """
        if len(image_or_files) < 1:
            return np.array([])

        # 检测第一个元素的类型
        if isinstance(image_or_files[0], Image.Image):

            def func(x):
                return x

        elif isinstance(image_or_files[0], str):

            def func(x):
                return Image.open(x)

        else:
            raise TypeError("image_or_file must be Image or str")

        part_features = []
        with torch.no_grad():
            process_bar = tqdm(range(0, len(image_or_files)), desc="batch encode image")
            for i in range(0, len(image_or_files), batch_size):
                # 图片预处理
                cur_batch = image_or_files[i : i + batch_size]
                cur_batch = [func(x) for x in cur_batch]
                # 感觉这个预处理也好慢
                cur_batch = [self.image_preprocess(x) for x in cur_batch]
                cur_batch = torch.stack(cur_batch, dim=0).to(self.device)
                image_features = self.model.encode_image(cur_batch)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                part_features.append(image_features.cpu().numpy())
                process_bar.update(batch_size)
        all_image_features = np.concatenate(part_features, axis=0)
        return all_image_features


if __name__ == "__main__":
    model_path = r"G:\code\github\Chinese-CLIP\datapath\experiments\muge_finetune_vit-b-16_roberta-base_bs160\checkpoints\epoch_latest.pt"
    clip_model = ClipModel(model_path)
    print(clip_model.device)
    text = "这是一只猫"
    image_path = r"G:\code\github\Chinese-CLIP\examples\pokemon.jpeg"
    text_features = clip_model.encode_text(text)
    print(text_features.shape)
    image_features = clip_model.encode_image(image_path)
    print(image_features.shape)
