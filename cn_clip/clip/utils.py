# Code modified from https://github.com/openai/CLIP

import json
import os
from pathlib import Path
from typing import Union, List
import urllib

import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode
from tqdm import tqdm

from cn_clip.clip import _tokenizer
from cn_clip.clip.model import convert_weights, CLIP, restore_model

__all__ = ["load", "tokenize", "available_models", "image_transform", "load_from_name"]

# 模型种类和下载地址
_MODELS = {
    "ViT-B-16": "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-b-16.pt",
    "ViT-L-14": "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-l-14.pt",
    "ViT-L-14-336": "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-l-14-336.pt",
    "ViT-H-14": "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-h-14.pt",
    "RN50": "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_rn50.pt",
}
# 模型信息, struct 是模型结构, 用 @ 分隔, 左边是图片模型, 右边是文本模型, input_resolution 是输入图片的分辨率
_MODEL_INFO = {
    "ViT-B-16": {"struct": "ViT-B-16@RoBERTa-wwm-ext-base-chinese", "input_resolution": 224},
    "ViT-L-14": {"struct": "ViT-L-14@RoBERTa-wwm-ext-base-chinese", "input_resolution": 224},
    "ViT-L-14-336": {"struct": "ViT-L-14-336@RoBERTa-wwm-ext-base-chinese", "input_resolution": 336},
    "ViT-H-14": {"struct": "ViT-H-14@RoBERTa-wwm-ext-large-chinese", "input_resolution": 224},
    "RN50": {"struct": "RN50@RBT3-chinese", "input_resolution": 224},
}


def _download(url: str, root: str):
    """
    下载到指定目录下
    """
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")), ncols=80, unit="iB", unit_scale=True, unit_divisor=1024
        ) as loop:
            while True:
                # 没想到是这样操作的, 每次读取 8192 字节, 即 8KB, 然后写入
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load_from_name(
    name: str,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    download_root: str = None,
    vision_model_name: str = None,
    text_model_name: str = None,
    input_resolution: int = None,
    convert_to_float16: bool = True,
):
    """
    基于名字加载模型
    name: 模型名字 或者 模型的路径
    device: cpu 或者 cuda
    download_root: 模型下载的根目录
    vision_model_name: 图片模型名字
    text_model_name: 文本模型名字
    input_resolution: 输入图片的分辨率
    convert_to_float16: 将模型精度转换为 float16
    """
    if name in _MODELS:
        # 只给名字, 就尝试下载模型
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
        model_name, model_input_resolution = _MODEL_INFO[name]["struct"], _MODEL_INFO[name]["input_resolution"]
    elif os.path.isfile(name):
        # 如果是文件, 就要求这三个参数都在 vision_model_name and text_model_name and input_resolution
        assert (
            vision_model_name and text_model_name and input_resolution
        ), "Please specify specific 'vision_model_name', 'text_model_name', and 'input_resolution'"
        model_path = name
        model_name, model_input_resolution = f"{vision_model_name}@{text_model_name}", input_resolution
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    # 读取检查点
    with open(model_path, "rb") as opened_file:
        # loading saved checkpoint
        checkpoint = torch.load(opened_file, map_location="cpu")

    # 创建模型
    model = create_model(model_name, checkpoint, convert_to_float16=convert_to_float16)
    # 将模型转换到指定设备
    if str(device) == "cpu":
        # 直接用 .float() 将模型精度转换为 float32
        model.float()
    else:
        model.to(device)
    # 返回模型和图片处理函数
    return model, image_transform(model_input_resolution)


def load(
    model,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    clip_path=None,
    bert_path=None,
    use_flash_attention=False,
):
    """Load CLIP and BERT model weights"""

    bert_state_dict = torch.load(bert_path, map_location="cpu") if bert_path else None
    clip_state_dict = torch.load(clip_path, map_location="cpu") if clip_path else None

    restore_model(model, clip_state_dict, bert_state_dict, use_flash_attention).to(device)

    if str(device) == "cpu":
        model.float()
    return model


def tokenize(texts: Union[str, List[str]], context_length: int = 52) -> torch.LongTensor:
    """
    分词函数, 具体的功能是由 _tokenizer 完成的
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all baseline models use 52 as the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    all_tokens = []
    for text in texts:
        all_tokens.append(
            [_tokenizer.vocab["[CLS]"]]
            + _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(text))[: context_length - 2]
            + [_tokenizer.vocab["[SEP]"]]
        )

    # zero padding, 第一个元素就是 [PAD]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        assert len(tokens) <= context_length
        result[i, : len(tokens)] = torch.tensor(tokens)

    # 返回的是 tensor, shape = [number of input strings, context_length]
    return result


def _convert_to_rgb(image):
    return image.convert("RGB")


def image_transform(image_size=224):
    transform = Compose(
        [
            Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )
    return transform


def create_model(model_name, checkpoint=None, convert_to_float16=True):
    """
    创建模型, 并从 checkpoint 中加载权重
    model_name: 模型名字
    checkpoint: 检查点
    convert_to_float16: 是否将模型精度转换为 float16
    """
    vision_model, text_model = model_name.split("@")
    # Initialize the model. 读取配置文件
    vision_model_config_file = Path(__file__).parent / f"model_configs/{vision_model.replace('/', '-')}.json"
    print("Loading vision model config from", vision_model_config_file)
    assert os.path.exists(vision_model_config_file)

    text_model_config_file = Path(__file__).parent / f"model_configs/{text_model.replace('/', '-')}.json"
    print("Loading text model config from", text_model_config_file)
    assert os.path.exists(text_model_config_file)

    # 从配置文件中读取模型信息, 组建成 model_info
    with open(vision_model_config_file, "r") as fv, open(text_model_config_file, "r") as ft:
        model_info = json.load(fv)
        for k, v in json.load(ft).items():
            model_info[k] = v
    if isinstance(model_info["vision_layers"], str):
        model_info["vision_layers"] = eval(model_info["vision_layers"])
    print("Model info", model_info)

    # 初始化模型
    model = CLIP(**model_info)
    # 将模型精度转换为 float16
    if convert_to_float16:
        convert_weights(model)
    # 从检查点中加载权重
    if checkpoint:
        sd = checkpoint["state_dict"]
        # 如果 sd 的 key 是以 module. 开头的, 就去掉 module.
        if next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module.") :]: v for k, v in sd.items() if "bert.pooler" not in k}
        model.load_state_dict(sd)
    return model
