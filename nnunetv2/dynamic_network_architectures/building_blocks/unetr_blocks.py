import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from typing import Tuple, Union, List, Type

from nnunetv2.dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim


def _norm(channels: int,
          conv_op: Type[_ConvNd],
          norm_op: Type[nn.Module],
          norm_kwargs: dict):
    """
    Choose normalization: prefer norm_op if provided, otherwise instance norm based on conv_op dim.
    """
    if norm_op is not None:
        return norm_op(channels, **(norm_kwargs or {}))
    dim = convert_conv_op_to_dim(conv_op)
    return nn.InstanceNorm3d(channels) if dim == 3 else nn.InstanceNorm2d(channels)


class PatchEmbedND(nn.Module):
    """ND Patch Embedding via conv, flatten to tokens."""
    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 patch_size: Union[int, Tuple[int,...]],
                 conv_op: Type[_ConvNd],
                 bias: bool = False,
                 norm_op: Type[nn.Module] = None,
                 norm_kwargs: dict = None):
        super().__init__()
        # ensure patch_size tuple
        dim = convert_conv_op_to_dim(conv_op)
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * dim
        self.proj = conv_op(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias
        )
        self.norm = norm_op(embed_dim, **(norm_kwargs or {})) if norm_op else None
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int,...]]:
        # x: (B, C, *spatial)
        x = self.proj(x)
        if self.norm:
            x = self.norm(x)
        B, C, *sp = x.shape
        N = int(torch.prod(torch.tensor(sp)))
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x, tuple(sp)


class ViTNDEncoder(nn.Module):
    """ViT Encoder wrapper emitting skips at defined depths."""
    def __init__(self,
                 embed_dim: int,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 depths: Tuple[int, ...] = None):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        if depths is None:
            depths = (num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1)
        self.depths = set(depths)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        skips = []
        for i, layer in enumerate(self.encoder.layers):
            x = layer(x)
            if i in self.depths:
                skips.append(x)
        return skips


class ConvStackND(nn.Module):
    """Stack of conv→norm→nonlin→dropout blocks."""
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 conv_bias: bool = False,
                 norm_op: Type[nn.Module] = None,
                 norm_op_kwargs: dict = None,
                 norm_kwargs: dict = None,
                 dropout_op: Type[nn.Module] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Type[nn.Module] = None,
                 nonlin_kwargs: dict = None):
        super().__init__()
        layers = []
        for i in range(num_convs):
            inc = in_ch if i == 0 else out_ch
            layers.append(conv_op(
                inc,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=conv_bias
            ))
            if norm_op:
                layers.append(norm_op(out_ch, **(norm_kwargs or {})))
            if nonlin:
                layers.append(nonlin(**(nonlin_kwargs or {})))
            if dropout_op:
                layers.append(dropout_op(**(dropout_op_kwargs or {})))
        self.stack = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)