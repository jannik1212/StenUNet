import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from typing import Union, Type, List, Tuple

from nnunetv2.dynamic_network_architectures.building_blocks.helper import (
    convert_conv_op_to_dim,
    get_matching_pool_op
)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int, conv_op: Type[_ConvNd], norm_op: Type[nn.Module]):
        super().__init__()
        self.W_g = nn.Sequential(
            conv_op(F_g, F_int, kernel_size=1, bias=True),
            norm_op(F_int)
        )
        self.W_x = nn.Sequential(
            conv_op(F_l, F_int, kernel_size=1, bias=True),
            norm_op(F_int)
        )
        self.psi = nn.Sequential(
            conv_op(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ConvStack(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 norm_op: Type[nn.Module],
                 norm_op_kwargs: dict,
                 dropout_op: Type[_DropoutNd],
                 dropout_op_kwargs: dict,
                 nonlin: Type[nn.Module],
                 nonlin_kwargs: dict):
        super().__init__()
        layers = []
        for i in range(num_convs):
            ic = in_channels if i == 0 else out_channels
            layers.append(conv_op(ic, out_channels, kernel_size=3, padding=1, bias=False))
            if norm_op is not None:
                layers.append(norm_op(out_channels, **(norm_op_kwargs or {})))
            if nonlin is not None:
                layers.append(nonlin(**(nonlin_kwargs or {})))
            if dropout_op is not None:
                layers.append(dropout_op(**(dropout_op_kwargs or {})))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Type[nn.Module] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Type[_DropoutNd] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Type[nn.Module] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 pool: str = 'conv'  # 'conv', 'max' or 'avg'
                 ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes

        # broadcast to lists
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        assert len(kernel_sizes) == n_stages
        assert len(strides) == n_stages
        assert len(features_per_stage) == n_stages
        assert len(n_conv_per_stage) == n_stages

        dim = convert_conv_op_to_dim(conv_op)

        # Encoder
        self.encoders = nn.ModuleList()
        self.pooling = nn.ModuleList()
        in_ch = input_channels
        for s in range(n_stages):
            # conv‐stack
            self.encoders.append(
                ConvStack(in_ch,
                          features_per_stage[s],
                          n_conv_per_stage[s],
                          conv_op, norm_op, norm_op_kwargs,
                          dropout_op, dropout_op_kwargs,
                          nonlin, nonlin_kwargs)
            )
            # pooling
            if pool in ('max', 'avg'):
                PoolOp = get_matching_pool_op(conv_op, pool_type=pool)
                self.pooling.append(PoolOp(kernel_size=strides[s], stride=strides[s]))
            else:  # 'conv'
                self.pooling.append(conv_op(in_channels=features_per_stage[s],
                                            out_channels=features_per_stage[s],
                                            kernel_size=strides[s],
                                            stride=strides[s],
                                            bias=conv_bias))
            in_ch = features_per_stage[s]

        # Bottleneck
        self.bottleneck = ConvStack(in_ch,
                                    features_per_stage[-1],
                                    n_conv_per_stage[-1],
                                    conv_op, norm_op, norm_op_kwargs,
                                    dropout_op, dropout_op_kwargs,
                                    nonlin, nonlin_kwargs)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        for d in range(n_stages - 2, -1, -1):
            # upsample
            if dim == 3:
                self.upconvs.append(
                    nn.ConvTranspose3d(features_per_stage[d+1],
                                       features_per_stage[d],
                                       kernel_size=strides[d+1],
                                       stride=strides[d+1])
                )
            else:
                self.upconvs.append(
                    nn.ConvTranspose2d(features_per_stage[d+1],
                                       features_per_stage[d],
                                       kernel_size=strides[d+1],
                                       stride=strides[d+1])
                )
            
            # attention gate
            self.attentions.append(
                AttentionBlock(
                    F_g=features_per_stage[d],   # decoder input x
                    F_l=features_per_stage[d+1],     # skip connection
                    F_int=min(features_per_stage[d+1], features_per_stage[d]) // 2,
                    conv_op=conv_op,
                    norm_op=norm_op
                )
            )
            # conv‐stack decoder
            in_channels = features_per_stage[d+1] + features_per_stage[d]
            self.decoders.append(
                ConvStack(in_channels,
                          features_per_stage[d],
                          n_conv_per_stage_decoder[d],
                          conv_op, norm_op, norm_op_kwargs,
                          dropout_op, dropout_op_kwargs,
                          nonlin, nonlin_kwargs)
            )

            # segmentation head
            self.seg_layers.append(conv_op(features_per_stage[d],
                                           num_classes,
                                           kernel_size=1))

        # dummy decoder for deep_supervision flag
        self.decoder = nn.Module()
        self.decoder.deep_supervision = deep_supervision

        # stash for VRAM estimator
        self._features = features_per_stage
        self._strides = strides

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = []
        for enc, pool in zip(self.encoders, self.pooling):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        segs = []
        for up, attn, dec, seg in zip(self.upconvs, self.attentions,
                                      self.decoders, self.seg_layers):
            x = up(x)
            skip = skips.pop()
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x,
                                  size=skip.shape[2:],
                                  mode='trilinear' if x.dim()==5 else 'bilinear',
                                  align_corners=False)
            skip = attn(x, skip)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)
            segs.append(seg(x))
        segs = segs[::-1]
        if not self.deep_supervision:
            return segs[0]
        return segs

    def compute_conv_feature_map_size(self, input_size: Tuple[int, ...]) -> int:
        """
        Multiply feature‐map spatial volume by channel counts at each stage,
        plus bottleneck and decoder + seg‐head maps.
        """
        sizes = list(input_size)
        total = 0

        # encoder
        for i, f in enumerate(self._features):
            total += f * np.prod(sizes)
            if i < len(self._features)-1:
                for d in range(len(sizes)):
                    sizes[d] //= self._strides[i+1]

        # bottleneck
        total += self._features[-1] * np.prod(sizes)

        # decoder + seg heads
        for i, f in enumerate(reversed(self._features[:-1])):
            for d in range(len(sizes)):
                sizes[d] *= self._strides[-(i+1)]
            total += f * np.prod(sizes)
            total += self.num_classes * np.prod(sizes)

        return int(total)
