from collections import OrderedDict
from typing import Tuple, Union
from itertools import repeat
import collections.abc

import math
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast

import importlib.util

if importlib.util.find_spec("flash_attn"):
    FlashMHA = importlib.import_module("flash_attn.flash_attention").FlashMHA

from cn_clip.clip import _tokenizer
from cn_clip.clip.configuration_bert import BertConfig
from cn_clip.clip.modeling_bert import BertModel


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        # FIXME support for non-transformer
        pass

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    残差注意力块, 单层网络, 表示 transformer 中的一个 block
    """

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, use_flash_attention: bool = False):
        """
        假设 d_model = 768, n_head = 12
        """
        super().__init__()

        # 多头注意力层. 可以看看 pytorch 的实现
        self.attn = nn.MultiheadAttention(d_model, n_head) if not use_flash_attention else FlashMHA(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    # 先从 768 -> 3072, 然后再从 3072 -> 768
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.use_flash_attention = use_flash_attention

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if self.use_flash_attention:
            # Batch first is needed for FlashAttention. See https://github.com/HazyResearch/flash-attention/issues/84 for more information.
            return self.attn(x.transpose(1, 0))[0].transpose(1, 0)
        else:
            # 先看这里. 他的输入是 (query, key, value), 自注意力机制中, query, key, value 都是同一个 x
            # self.attn 的返回值是 (attn_output, attn_output_weights)
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        """
        前向传播
        x 的 shape 是  (197, batch_size, 768).
        """
        # x + 是指残差连接. ln_1 会先对输入 x 进行归一化, 然后进入 attention 层.
        # nn.MultiheadAttention 的输入输出的 shape 是一样的
        x = x + self.attention(self.ln_1(x))
        # mlp 是一个两层的全连接网络, 用于对 attention 层的输出进行处理
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, use_flash_attention: bool = False
    ):
        """
        假设 width = 768, layers = 12, heads = 12
        """
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        # 构建 12 层的 transformer
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask, use_flash_attention) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        """
        x 的 shape 是  (197, batch_size, 768).
        """
        # 使用检查点技术, 降低显存占用, 但是会加重计算量
        if self.grad_checkpointing and not torch.jit.is_scripting():
            # 只要遍历模型中的每一层, 然后对每一层进行检查点
            for r in self.resblocks:
                x = checkpoint(r, x)
            return x
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    """
    看一看吧, 都还没看过 vit
    """

    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        use_flash_attention: bool = False,
    ):
        """
        假设 input_resolution = 224, patch_size = 16, width = 768, layers = 12, heads = 12, output_dim = 512
        """
        super().__init__()
        self.input_resolution = input_resolution
        # grid_size = (14, 14) 网格数
        self.grid_size = (self.input_resolution // patch_size, self.input_resolution // patch_size)
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # scale = 1 / sqrt(768)
        scale = width**-0.5
        # class_embedding 的 shape 是 (768, ). class_embedding 是第一个 token, 用来表示图片的类别
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        # positional_embedding 的 shape 是 (196 + 1, 768)
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, use_flash_attention=use_flash_attention)

        self.ln_post = LayerNorm(width)
        # proj 的 shape 是 (768, 512)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """
        启用梯度检查点技术
        """
        # 只要设置个属性就行了
        self.transformer.grad_checkpointing = enable

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int((L - 1) * (1 - mask_ratio))

        noise = torch.rand(N, L - 1, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1) + torch.ones(N, L - 1, device=x.device, dtype=int)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        x0 = x[:, 0, :]
        x0 = x0.reshape(N, 1, D)
        x_masked_add = torch.cat([x0, x_masked], axis=1)
        return x_masked_add

    # @autocast()  # 会使得图片的输出精度变成 float16
    def forward(self, x: torch.Tensor, mask_ratio: float = 0.0):
        """
        看看前向传播的过程
        x 的 shape 是 (batch_size, 3, 224, 224)
        """
        # x 的 shape 是 (batch_size, 768, 14, 14)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # x 的 shape 是 (batch_size, 768, 196)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        # x 的 shape 是 (batch_size, 196, 768)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # x 的 shape 是 (batch_size, 197, 768). 这是在叠加第一个位置的 token
        # self.class_embedding 的 shape 是 (768,), 通过叠加一个 zeros 的 (batch_size, 1, 768) 的 tensor, 来复制 class_embedding
        # 就变成了 (batch_size, 1, 768) + (batch_size, 196, 768) = (batch_size, 197, 768
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        # x 的 shape 是 (batch_size, 197, 768). 添加位置编码
        x = x + self.positional_embedding.to(x.dtype)
        # x 的 shape 是 (batch_size, 197, 768), 是怎么理解的, 197 是序列长度, 768 是特征维度, 类比于 bert
        # TODO: 随机掩码, 没看
        if mask_ratio != 0:
            x = self.random_masking(x, mask_ratio)
        # 经过 LayerNorm
        x = self.ln_pre(x)

        # x 的 shape 是  (197, batch_size, 768). 这是 nn.MultiheadAttention 要求的
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        # x 的 shape 是 (batch_size, 197, 768)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x 的 shape 是 (batch_size, 768). 只对第一个 token 做了 LayerNorm
        x = self.ln_post(x[:, 0, :])

        # x 的 shape 是 (batch_size, 512). 做一个线性变换
        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    """
    这里记录的维度以 ViT-B-16 和 RoBERTa-wwm-ext-base-chinese 为例.
    """

    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        # text
        vocab_size: int,
        text_attention_probs_dropout_prob: float,
        text_hidden_act: str,
        text_hidden_dropout_prob: float,
        text_hidden_size: int,
        text_initializer_range: float,
        text_intermediate_size: int,
        text_max_position_embeddings: int,
        text_num_attention_heads: int,
        text_num_hidden_layers: int,
        text_type_vocab_size: int,
        tokenizer=_tokenizer,
        # vision head width, added this param for ViT-H
        vision_head_width: int = 64,
        use_flash_attention: bool = False,
    ):
        super().__init__()

        # 初始化视觉模型
        # resnet 的先不看
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // vision_head_width
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
            )
        else:
            vision_heads = vision_width // vision_head_width  # 768 / 64 = 12
            self.visual = VisualTransformer(
                input_resolution=image_resolution,  # 224
                patch_size=vision_patch_size,  # 16
                width=vision_width,  # 768
                layers=vision_layers,  # 12
                heads=vision_heads,  # 12
                output_dim=embed_dim,  # 512
                use_flash_attention=use_flash_attention,
            )

        # 初始化语言模型
        self.bert_config = BertConfig(
            vocab_size_or_config_json_file=vocab_size,  # 21128
            hidden_size=text_hidden_size,  # 768
            num_hidden_layers=text_num_hidden_layers,  # 12
            num_attention_heads=text_num_attention_heads,  # 12
            intermediate_size=text_intermediate_size,  # 3072
            hidden_act=text_hidden_act,  # gelu
            hidden_dropout_prob=text_hidden_dropout_prob,  # 0.1
            attention_probs_dropout_prob=text_attention_probs_dropout_prob,  # 0.1
            max_position_embeddings=text_max_position_embeddings,  # 512
            type_vocab_size=text_type_vocab_size,  # 2
            initializer_range=text_initializer_range,  #  0.02
            layer_norm_eps=1e-12,
            use_flash_attention=use_flash_attention,
        )
        self.bert = BertModel(self.bert_config)

        # 文本投射层, 768 => 512
        self.text_projection = nn.Parameter(torch.empty(text_hidden_size, embed_dim))
        # logits 缩放, 是个可学习的参数. 是个温度参数. 初始是 2.659
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 分词器
        self.tokenizer = tokenizer

        # 初始化参数
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        初始化参数
        """
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features**-0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        # 初始化文本投射层
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.bert_config.hidden_size**-0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """
        启用梯度检查点
        """
        self.visual.set_grad_checkpointing(enable)
        self.bert.set_grad_checkpointing(enable)

    @property
    def dtype(self):
        """
        注意这里使用的是 dtype
        """
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, mask_ratio=0):
        """
        编码图片
        """
        if isinstance(self.visual, ModifiedResNet):
            # mask_ratio > 0 (FLIP strategy) is currently only implemented for VisualTransformer.
            return self.visual(image.type(self.dtype))
        return self.visual(image.type(self.dtype), mask_ratio)

    def encode_text(self, text):
        """
        编码文本
        """
        pad_index = self.tokenizer.vocab["[PAD]"]
        # 获取 attention mask. text.ne 应该是指 text != pad_index 的都是 1
        attn_mask = text.ne(pad_index).type(self.dtype)
        x = self.bert(text, attention_mask=attn_mask)[0].type(self.dtype)  # [batch_size, seq_length, hidden_size]
        # x[:, 0, :] 是取第一个 token 的输出, shape 是 [batch_size, hidden_size]. @ 是矩阵乘法
        # 最终的 shape 是 [batch_size, embed_dim]
        return x[:, 0, :] @ self.text_projection

    def forward(self, image, text, mask_ratio=0):
        """
        看下前向传播的过程
        image shape 是 (batch_size, 3, 224, 224)
        text shape 是 (batch_size, seq_length)  seq_length 默认是 52
        """
        assert image is not None or text is not None, "text and image cannot both be None!"

        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)
        image_features = self.encode_image(image, mask_ratio)
        text_features = self.encode_text(text)

        # 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 返回的是 logit_scale 的指数, 也就是 e^logit_scale
        return image_features, text_features, self.logit_scale.exp()

    def get_similarity(self, image, text):
        """
        获取相似度
        image shape 是 (batch_size, 3, 224, 224)
        text shape 是 (batch_size, seq_length)  seq_length 默认是 52
        """
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        # shape 是 (image_batch_size, text_batch_size)
        # 每一行是一张图片, 意思就是一个图片和所有的文本的相似度
        logits_per_image = logit_scale * image_features @ text_features.t()
        # shape 是 (text_batch_size, image_batch_size)
        # 每一行是一个文本, 意思就是一个文本和所有的图片的相似度
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_models_to_fp32(model: nn.Module):
    """
    这里还有个将模型精度转换成 float32 的函数
    """
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def convert_weights(model: CLIP):
    """
    将模型权重的精度转换成 float16
    Convert applicable model parameters to fp16"""

    print("将模型转换成 float16 精度")

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        if isinstance(l, BertModel):
            l.to(torch.half)

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def restore_model(model: CLIP, clip_state_dict: dict, bert_state_dict: dict, use_flash_attention: bool):
    """
    恢复模型
    """
    merged_state_dict = {}

    # 读取 clip 的 vit 部分的权重
    # use clip_state_dict to initialize the image encoder & logit scale
    if clip_state_dict is not None:
        for k, v in clip_state_dict.items():
            if k.startswith("visual") or k == "logit_scale":
                merged_state_dict[k] = v

    # 读取 bert 的权重
    # use bert_state_dict to initialize the text encoder
    if bert_state_dict is not None:
        for k, v in bert_state_dict.items():
            if k.startswith("bert") and "bert.pooler" not in k:
                merged_state_dict[k] = v

    # adapt flash attention
    if use_flash_attention:
        merged_state_dict = convert_state_dict(merged_state_dict)

    # 将模型的权重转换成 float16
    convert_weights(model)
    # 重置位置嵌入
    resize_pos_embed(merged_state_dict, model)
    # 将模型的权重加载到模型中
    model.load_state_dict(merged_state_dict, strict=False)
    return model.eval()


def convert_state_dict(state_dict):
    """Adapt to Flash Attention
    适应 Flash Attention
    """
    if not state_dict:
        return state_dict

    # 定义前缀, 兼容多卡训练的保存结果
    prefix = "module." if list(state_dict.keys())[0].startswith("module") else ""

    if f"{prefix}visual.transformer.resblocks.0.attn.in_proj_weight" in state_dict:
        for k in list(state_dict.keys()):
            # 更换名字
            if "attn.in_proj_weight" in k:
                state_dict[k.replace("attn.in_proj_weight", "attn.Wqkv.weight")] = state_dict.pop(k)
            elif "attn.in_proj_bias" in k:
                state_dict[k.replace("attn.in_proj_bias", "attn.Wqkv.bias")] = state_dict.pop(k)
    elif f"{prefix}visual.transformer.resblocks.0.attn.Wqkv.weight" in state_dict:
        for k in list(state_dict.keys()):
            # 同样时更换名字, 和上面的是相反的
            if "attn.Wqkv.weight" in k:
                state_dict[k.replace("attn.Wqkv.weight", "attn.in_proj_weight")] = state_dict.pop(k)
            elif "attn.Wqkv.bias" in k:
                state_dict[k.replace("attn.Wqkv.bias", "attn.in_proj_bias")] = state_dict.pop(k)

    if f"{prefix}bert.encoder.layer.0.attention.self.query.weight" in state_dict:
        i = 0
        while f"{prefix}bert.encoder.layer.{i}.attention.self.query.weight" in state_dict:
            state_dict[f"{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.weight"] = torch.cat(
                (
                    state_dict.pop(f"{prefix}bert.encoder.layer.{i}.attention.self.query.weight"),
                    state_dict.pop(f"{prefix}bert.encoder.layer.{i}.attention.self.key.weight"),
                    state_dict.pop(f"{prefix}bert.encoder.layer.{i}.attention.self.value.weight"),
                )
            )
            state_dict[f"{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.bias"] = torch.cat(
                (
                    state_dict.pop(f"{prefix}bert.encoder.layer.{i}.attention.self.query.bias"),
                    state_dict.pop(f"{prefix}bert.encoder.layer.{i}.attention.self.key.bias"),
                    state_dict.pop(f"{prefix}bert.encoder.layer.{i}.attention.self.value.bias"),
                )
            )
            state_dict[f"{prefix}bert.encoder.layer.{i}.attention.self.out_proj.weight"] = state_dict.pop(
                f"{prefix}bert.encoder.layer.{i}.attention.output.dense.weight"
            )
            state_dict[f"{prefix}bert.encoder.layer.{i}.attention.self.out_proj.bias"] = state_dict.pop(
                f"{prefix}bert.encoder.layer.{i}.attention.output.dense.bias"
            )
            i += 1
    elif f"{prefix}bert.encoder.layer.0.attention.self.Wqkv.weight" in state_dict:
        i = 0
        # 果然这里也是相反的操作
        while f"{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.weight" in state_dict:
            (
                state_dict[f"{prefix}bert.encoder.layer.{i}.attention.self.query.weight"],
                state_dict[f"{prefix}bert.encoder.layer.{i}.attention.self.key.weight"],
                state_dict[f"{prefix}bert.encoder.layer.{i}.attention.self.value.weight"],
            ) = torch.chunk(state_dict.pop(f"{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.weight"), chunks=3)
            (
                state_dict[f"{prefix}bert.encoder.layer.{i}.attention.self.query.bias"],
                state_dict[f"{prefix}bert.encoder.layer.{i}.attention.self.key.bias"],
                state_dict[f"{prefix}bert.encoder.layer.{i}.attention.self.value.bias"],
            ) = torch.chunk(state_dict.pop(f"{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.bias"), chunks=3)
            state_dict[f"{prefix}bert.encoder.layer.{i}.attention.output.dense.weight"] = state_dict.pop(
                f"{prefix}bert.encoder.layer.{i}.attention.self.out_proj.weight"
            )
            state_dict[f"{prefix}bert.encoder.layer.{i}.attention.output.dense.bias"] = state_dict.pop(
                f"module.bert.encoder.layer.{i}.attention.self.out_proj.bias"
            )
            i += 1

    return state_dict


def resize_pos_embed(state_dict, model: CLIP, interpolation: str = "bicubic", seq_dim=1, prefix=""):
    """
    重置位置嵌入
    """
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get(prefix + "visual.positional_embedding", None)
    # 找到原始模型
    model = model.module if hasattr(model, "module") else model
    if old_pos_embed is None or not hasattr(model.visual, "grid_size"):
        return
    # to_2tuple 将输入转换为长度为2的元组. grid_size 是网格数, 就是图片被切成了多少块
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    # 总的序列长度
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    # 如果长度一致, 就不用更改了, 直接返回
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    # 旧的网格大小
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    # 缩放位置嵌入
    logging.info("Resizing position embedding grid-size from %s to %s", old_grid_size, grid_size)
    # pos_emb_img 的 shape 是 (1, height, width, channel) => (1, channel, height, width)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    # 这就是缩放函数
    pos_emb_img = F.interpolate(
        pos_emb_img,  # 输入
        size=grid_size,  # 缩放到的大小
        mode=interpolation,  # 插值方式
        align_corners=True,  # 缩放时保证角点像素不变
    )
    # (1, channel, height, width) => (1, height, width, channel) => (1, height * width, channel) => (height * width, channel)
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict[prefix + "visual.positional_embedding"] = new_pos_embed


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        # 将 x 重复 n 次
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)
