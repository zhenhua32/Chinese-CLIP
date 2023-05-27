"""
实现搜索功能
"""

import base64
import io
import json
import os

import faiss
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from model import ClipModel, base64_to_image

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 全局文件
image_vectors_file = r"G:\code\github\Chinese-CLIP\datapath\datasets\MUGE\image_vectors.npy"
index2image_file = r"G:\code\github\Chinese-CLIP\datapath\datasets\MUGE\index2image.json"
model_path = r"G:\code\github\Chinese-CLIP\datapath\experiments\muge_finetune_vit-b-16_roberta-base_bs160\checkpoints\epoch_latest.pt"
query_files = [
    r"G:\code\github\Chinese-CLIP\datapath\datasets\MUGE\train_texts.jsonl",
    r"G:\code\github\Chinese-CLIP\datapath\datasets\MUGE\valid_texts.jsonl",
]


class Searcher:
    def __init__(self) -> None:
        self.model = self.load_model()
        self.index_engine, self.index2image, self.image_id2index = self.load_search_engine()
        self.query2image_ids = self.load_query_info()

    def load_image_vectors(self):
        """
        加载图片向量
        """
        image_vectors = np.load(image_vectors_file)
        return image_vectors

    def load_image_info(self):
        """
        加载图片信息
        返回
        index2image: dict, index -> image_info, {image_id, image_base64}
        image_id2index: dict, image_id -> index
        """
        with open(index2image_file, "r", encoding="utf-8") as f:
            index2image = json.load(f)

        image_id2index = dict()
        for index, info in index2image.items():
            image_id = info["image_id"]
            image_id2index[image_id] = index

        return index2image, image_id2index

    def load_search_engine(self):
        """
        加载搜索引擎
        """
        image_vectors = self.load_image_vectors()
        index2image, image_id2index = self.load_image_info()

        index_engine = faiss.IndexFlatIP(image_vectors.shape[-1])
        index_engine.add(image_vectors)
        return index_engine, index2image, image_id2index

    def load_query_info(self):
        """
        加载已有标注的query
        """
        query2image_ids = dict()
        for file in query_files:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    line = json.loads(line)
                    query = line["text"]
                    image_ids = line["image_ids"]
                    query2image_ids[query] = image_ids
        return query2image_ids

    def load_model(self):
        """
        加载模型
        """
        clip_model = ClipModel(model_path)
        return clip_model

    def search(self, query: str, top_k: int = 10):
        """
        对 query 进行搜索
        """
        # 获取query的向量
        query_vector = self.model.encode_text(query)
        query_vector = query_vector.reshape(1, -1).astype(np.float32)

        distance_matrix, index_matrix = self.index_engine.search(query_vector, top_k)
        return distance_matrix[0], index_matrix[0]

    def display_images(self, title, index_list, gold_ids=None, top_k: int = 10):
        """
        index_list 是通过向量查找的图片index, list[int]
        gold_ids 是实际标签的图片id, list[int]
        """
        # 创建一个 2 行 k 列的画布
        fig, ax = plt.subplots(2, top_k, figsize=(20, 5))
        # 第一行展示实际标签图片
        if gold_ids:
            for i, image_id in enumerate(gold_ids):
                index = self.image_id2index[str(image_id)]
                image_base64 = self.index2image[index]["image_base64"]
                image = base64_to_image(image_base64)
                ax[0, i].imshow(image)
                ax[0, i].set_title(image_id)
        # 第二行展示检索结果
        for i, index in enumerate(index_list):
            image_id = self.index2image[str(index)]["image_id"]
            image_base64 = self.index2image[str(index)]["image_base64"]
            image = base64_to_image(image_base64)
            ax[1, i].imshow(image)
            ax[1, i].set_title(image_id)
        # 隐藏坐标轴
        for a in ax.flat:
            a.axis("off")
        plt.suptitle(f"query: {title}. 第一行为实际标签图片, 第二行为检索结果")
        plt.tight_layout()
        # 显示画布
        # plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer)

        # 从流中读取图片并创建一个 PIL 对象
        buffer.seek(0)  # 移动到流的开始位置
        pil_image = Image.open(buffer)

        return pil_image
