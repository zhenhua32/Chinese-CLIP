from math import ceil
import os
import logging
from pathlib import Path
import json
from PIL import Image
import base64
from io import BytesIO
from dataclasses import dataclass

import lmdb
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from timm.data import create_transform

from cn_clip.clip import _tokenizer
from cn_clip.clip import tokenize


def _convert_to_rgb(image):
    """图片转换为 RGB 格式"""
    return image.convert("RGB")


def _preprocess_text(text):
    """文本小写后替换中文引号为英文引号"""
    # adapt the text to Chinese BERT vocab
    text = text.lower().replace("“", '"').replace("”", '"')
    return text


class LMDBDataset(Dataset):
    """
    看下数据集是怎么构造的
    """

    def __init__(self, lmdb_path, split="val", max_txt_length=64, use_augment=False, resolution=224):
        self.lmdb_path = lmdb_path

        # 分别加载 imgs 目录和 pairs 目录
        # assert LMDB directories exist
        assert os.path.isdir(lmdb_path), "The LMDB directory {} of {} split does not exist!".format(lmdb_path, split)
        lmdb_pairs = os.path.join(lmdb_path, "pairs")
        assert os.path.isdir(lmdb_pairs), "The LMDB directory {} of {} image-text pairs does not exist!".format(
            lmdb_pairs, split
        )
        lmdb_imgs = os.path.join(lmdb_path, "imgs")
        assert os.path.isdir(lmdb_imgs), "The LMDB directory {} of {} image base64 strings does not exist!".format(
            lmdb_imgs, split
        )

        # open LMDB files
        self.env_pairs = lmdb.open(lmdb_pairs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_pairs = self.env_pairs.begin(buffers=True)
        self.env_imgs = lmdb.open(lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_imgs = self.env_imgs.begin(buffers=True)

        # 获取数量
        # fetch number of pairs and images
        self.number_samples = int(self.txn_pairs.get(key=b"num_samples").tobytes().decode("utf-8"))
        self.number_images = int(self.txn_imgs.get(key=b"num_images").tobytes().decode("utf-8"))
        logging.info(
            "{} LMDB file contains {} images and {} pairs.".format(split, self.number_images, self.number_samples)
        )

        super(LMDBDataset, self).__init__()

        # 这两个参数都是会在后续被更新的
        # the self.dataset_len will be edited to a larger value by calling pad_dataset()
        self.dataset_len = self.number_samples
        self.global_batch_size = 1  # will be modified to the exact global_batch_size after calling pad_dataset()

        # 数据集的类型
        self.split = split
        # 文本的最大长度
        self.max_txt_length = max_txt_length

        # 是否使用数据增强
        self.use_augment = use_augment
        self.transform = self._build_transform(resolution)

    def _build_transform(self, resolution):
        if self.split == "train" and self.use_augment:
            # 当使用数据增强时
            transform = create_transform(
                input_size=resolution,
                scale=(0.9, 1.0),
                is_training=True,
                color_jitter=None,
                auto_augment="original",
                interpolation="bicubic",
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            )
            transform = Compose(transform.transforms[:-3] + [_convert_to_rgb] + transform.transforms[-3:])
        else:
            # 没有数据增强时, 只有一个缩放和归一化
            transform = Compose(
                [
                    Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
                    _convert_to_rgb,
                    ToTensor(),
                    # 这里三个参数是表示 RGB 的三个通道的均值和标准差
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ]
            )
        return transform

    def __del__(self):
        """
        关闭 lmdb 文件
        """
        if hasattr(self, "env_pairs"):
            self.env_pairs.close()
        if hasattr(self, "env_imgs"):
            self.env_imgs.close()

    def __len__(self):
        """
        数据集的长度
        """
        return self.dataset_len

    def __getitem__(self, index):
        """
        根据索引获取数据
        """
        sample_index = index % self.number_samples

        # 加载数据
        pair = pickle.loads(self.txn_pairs.get("{}".format(sample_index).encode("utf-8")).tobytes())
        image_id, text_id, raw_text = pair

        # 根据 image_id 获取图片
        image_b64 = self.txn_imgs.get("{}".format(image_id).encode("utf-8")).tobytes()
        image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64)))  # already resized
        image = self.transform(image)

        # 分词. 获取的是第一个元素. tokenize 是用来批量处理文本的
        text = tokenize([_preprocess_text(raw_text)], context_length=self.max_txt_length)[0]
        # eos_index 是 [SEP] 的索引
        eos_index = text.numpy().tolist().index(_tokenizer.vocab["[SEP]"])
        return image, text, eos_index


class LMDBDatasetForWindows(Dataset):
    """
    windows 真的坑, 不能使用多线程读取 lmdb 文件. 果然还是 linux 先进点.
    报错是 TypeError: cannot pickle 'Environment' object
    https://github.com/pytorch/vision/issues/689#issuecomment-787215916

    # 出处是这个, 文档里有写, windows 上要使用 pickle 在进程间传递数据

    This separate serialization means that you should take two steps to ensure you are compatible with Windows while using multi-process data loading:

    Wrap most of you main script’s code within if __name__ == '__main__': block, to make sure it doesn’t run again (most likely generating error) when each worker process is launched. You can place your dataset and DataLoader instance creation logic here, as it doesn’t need to be re-executed in workers.

    Make sure that any custom collate_fn, worker_init_fn or dataset code is declared as top level definitions, outside of the __main__ check. This ensures that they are available in worker processes. (this is needed since functions are pickled as references only, not bytecode.)
    """

    def __init__(self, lmdb_path, split="val", max_txt_length=64, use_augment=False, resolution=224):
        self.lmdb_path = lmdb_path

        # 分别加载 imgs 目录和 pairs 目录
        # assert LMDB directories exist
        assert os.path.isdir(lmdb_path), "The LMDB directory {} of {} split does not exist!".format(lmdb_path, split)
        lmdb_pairs = os.path.join(lmdb_path, "pairs")
        assert os.path.isdir(lmdb_pairs), "The LMDB directory {} of {} image-text pairs does not exist!".format(
            lmdb_pairs, split
        )
        lmdb_imgs = os.path.join(lmdb_path, "imgs")
        assert os.path.isdir(lmdb_imgs), "The LMDB directory {} of {} image base64 strings does not exist!".format(
            lmdb_imgs, split
        )

        self.lmdb_pairs = lmdb_pairs
        self.lmdb_imgs = lmdb_imgs
        # 数据集的类型
        self.split = split
        # 文本的最大长度
        self.max_txt_length = max_txt_length

        # 唯一重要的是要定义 self.number_samples
        self._get_number_samples()

        super().__init__()

        # 这两个参数都是会在后续被更新的
        # the self.dataset_len will be edited to a larger value by calling pad_dataset()
        self.dataset_len = self.number_samples
        # self.dataset_len = self.number_samples
        self.global_batch_size = 1  # will be modified to the exact global_batch_size after calling pad_dataset()

        # 是否使用数据增强
        self.use_augment = use_augment
        self.transform = self._build_transform(resolution)

    def _build_transform(self, resolution):
        if self.split == "train" and self.use_augment:
            # 当使用数据增强时
            transform = create_transform(
                input_size=resolution,
                scale=(0.9, 1.0),
                is_training=True,
                color_jitter=None,
                auto_augment="original",
                interpolation="bicubic",
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            )
            transform = Compose(transform.transforms[:-3] + [_convert_to_rgb] + transform.transforms[-3:])
        else:
            # 没有数据增强时, 只有一个缩放和归一化
            transform = Compose(
                [
                    Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
                    _convert_to_rgb,
                    ToTensor(),
                    # 这里三个参数是表示 RGB 的三个通道的均值和标准差
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ]
            )
        return transform

    def _open_lmdb(self):
        # open LMDB files
        self.env_pairs = lmdb.open(
            self.lmdb_pairs, readonly=True, create=False, lock=False, readahead=False, meminit=False
        )
        self.txn_pairs = self.env_pairs.begin(buffers=True)
        self.env_imgs = lmdb.open(
            self.lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False
        )
        self.txn_imgs = self.env_imgs.begin(buffers=True)

    def _get_number_samples(self):
        env_pairs = lmdb.open(
            self.lmdb_pairs, readonly=True, create=False, lock=False, readahead=False, meminit=False
        )
        txn_pairs = env_pairs.begin(buffers=True)
        env_imgs = lmdb.open(
            self.lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False
        )
        txn_imgs = env_imgs.begin(buffers=True)
        # 获取数量
        # fetch number of pairs and images
        self.number_samples = int(txn_pairs.get(key=b"num_samples").tobytes().decode("utf-8"))
        self.number_images = int(txn_imgs.get(key=b"num_images").tobytes().decode("utf-8"))
        logging.info(
            "{} LMDB file contains {} images and {} pairs.".format(self.split, self.number_images, self.number_samples)
        )
        env_pairs.close()
        env_imgs.close()

    def __del__(self):
        """
        关闭 lmdb 文件
        """
        if hasattr(self, "env_pairs"):
            self.env_pairs.close()
        if hasattr(self, "env_imgs"):
            self.env_imgs.close()

    def __len__(self):
        """
        数据集的长度
        """
        return self.dataset_len

    def __getitem__(self, index):
        """
        根据索引获取数据
        """
        # 在第一次实际获取数据时, 打开 lmdb 文件
        if not hasattr(self, "txn_pairs"):
            self._open_lmdb()

        sample_index = index % self.number_samples

        # 加载数据
        pair = pickle.loads(self.txn_pairs.get("{}".format(sample_index).encode("utf-8")).tobytes())
        image_id, text_id, raw_text = pair

        # 根据 image_id 获取图片
        image_b64 = self.txn_imgs.get("{}".format(image_id).encode("utf-8")).tobytes()
        image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64)))  # already resized
        image = self.transform(image)

        # 分词. 获取的是第一个元素. tokenize 是用来批量处理文本的
        text = tokenize([_preprocess_text(raw_text)], context_length=self.max_txt_length)[0]
        # eos_index 是 [SEP] 的索引
        eos_index = text.numpy().tolist().index(_tokenizer.vocab["[SEP]"])
        return image, text, eos_index


def pad_dataset(dataset, global_batch_size):
    """
    填充 lmdb 数据集，使其可以被 global_batch_size 整除
    """
    # edit dataset.__len__() of the dataset
    dataset.dataset_len = ceil(dataset.dataset_len / global_batch_size) * global_batch_size
    dataset.global_batch_size = global_batch_size


def fetch_resolution(vision_model):
    """
    从模型的配置中读取图片的分辨率
    """
    # fetch the resolution from the vision model config
    vision_model_config_file = (
        Path(__file__).parent.parent / f"clip/model_configs/{vision_model.replace('/', '-')}.json"
    )
    with open(vision_model_config_file, "r") as fv:
        model_info = json.load(fv)
    return model_info["image_resolution"]


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler
    dataset: LMDBDataset
    epoch_id: int


def get_dataset(args, is_train, max_txt_length=64, epoch_id=0):
    """
    获取数据集
    args: 是一个命名空间，包含了所有的参数
    is_train: 是否是训练集
    max_txt_length: 文本的最大长度
    epoch_id: epoch 的顺序
    """
    if is_train:
        db_path = args.train_data
    else:
        db_path = args.val_data
    assert db_path is not None

    if os.name == "nt":
        print("Using LMDBDatasetForWindows")
        LMDBDataset = LMDBDatasetForWindows

    dataset = LMDBDataset(
        db_path,
        split="train" if is_train else "val",
        max_txt_length=max_txt_length,
        use_augment=args.use_augment if is_train else False,
        resolution=fetch_resolution(args.vision_model),
    )

    # pad the dataset splits using the beginning samples in the LMDB files
    # to make the number of samples enough for a full final global batch
    batch_size = args.batch_size if is_train else args.valid_batch_size
    # 总的 batch size
    global_batch_size = batch_size * torch.distributed.get_world_size()
    pad_dataset(dataset, global_batch_size)

    # 样本数
    num_samples = dataset.dataset_len
    # Update in 22.12.11: We have changed the **validation** dataset sampler during finetuning
    # from sequential to shuffled (in a determistic order between experiments and epochs).
    # This is to avoid there being one text matching multiple images (or vice versa) in a local batch
    # which will affect the correctness of computing the validation in-batch accuracy.
    # 这里将验证集的 shuffle 也设置成了 True
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)
    # 分布式采样器在每轮的时候要设置 epoch, 让每轮使用不同的采样顺序
    sampler.set_epoch(epoch_id if is_train else 0)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=args.num_workers if is_train else args.valid_num_workers,
        # num_workers=4,  # 1,2,4 都不错, 0 非常慢
        sampler=sampler,
    )

    # 需要更新样本数, 保证能整除. 并更新 batch 数量
    dataloader.num_samples = num_samples
    assert num_samples % dataset.global_batch_size == 0
    dataloader.num_batches = num_samples // dataset.global_batch_size

    # 这挺好, 返回 4 个字段
    return DataInfo(dataloader, sampler, dataset, epoch_id)


def get_data(args, epoch_id=0, max_txt_length=64):
    """
    加载数据集, 返回一个 dict, 里面有 train 和 val 两个 key, 分别对应训练集和验证集
    """
    data = {}

    if args.train_data:
        data["train"] = get_dataset(args, is_train=True, max_txt_length=max_txt_length, epoch_id=epoch_id)

    if args.val_data:
        data["val"] = get_dataset(args, is_train=False, max_txt_length=max_txt_length, epoch_id=epoch_id)

    return data
