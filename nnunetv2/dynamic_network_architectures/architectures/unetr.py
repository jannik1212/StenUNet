import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Union, Type

from nnunetv2.dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim, get_matching_convtransp
from nnunetv2.dynamic_network_architectures.building_blocks.unetr_blocks import (
    PatchEmbedND,
    ViTNDEncoder,
    ConvStackND
)


class UNETR(nn.Module):
    """Dynamic UNETR aligned with nnUNet pipeline signature."""
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[nn.Module],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Type[nn.Module] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Type[nn.Module] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Type[nn.Module] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.conv_op = conv_op
        self.num_classes = num_classes
        dim = convert_conv_op_to_dim(conv_op)
        assert dim in (2, 3), "UNETR only supports 2D or 3D."

        # normalize args
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * (n_stages - 1)
        if isinstance(strides, int):
            strides = [strides] * n_stages

        self.features_per_stage = list(features_per_stage)
        self.strides = strides

        # --- patch embedding ---
        base = self.features_per_stage[0]
        self.patch_embed = PatchEmbedND(
            input_channels,
            base,
            patch_size=strides[0],
            conv_op=conv_op,
            bias=conv_bias,
            norm_op=norm_op,
            norm_kwargs=norm_op_kwargs
        )
        self.pos_embed = None  # initialized on first forward

        # --- ViT encoder ---
        total_layers = sum(n_conv_per_stage)
        cum = np.cumsum(n_conv_per_stage)
        depths = tuple([int(c) - 1 for c in cum[:-1]]) if total_layers > 1 else ()
        num_heads = max(1, base // 64)
        self.vit = ViTNDEncoder(
            embed_dim=base,
            num_layers=total_layers,
            num_heads=num_heads,
            depths=depths
        )

        # --- decoder ---
        transp_op = get_matching_convtransp(conv_op)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.seg_heads = nn.ModuleList()

        for d in range(n_stages - 2, -1, -1):
            # upsample
            self.upconvs.append(
                transp_op(
                    self.features_per_stage[d + 1],
                    self.features_per_stage[d],
                    kernel_size=strides[d + 1],
                    stride=strides[d + 1],
                    bias=conv_bias
                )
            )
            # conv stack
            self.decoders.append(
                ConvStackND(
                    self.features_per_stage[d] * 2,
                    self.features_per_stage[d],
                    num_convs=n_conv_per_stage_decoder[d],
                    conv_op=conv_op,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs
                )
            )
            # segmentation head at this scale
            self.seg_heads.append(
                conv_op(
                    self.features_per_stage[d],
                    num_classes,
                    kernel_size=1,
                    bias=True
                )
            )

    def forward(self, x: torch.Tensor):
        # embed
        B, C, *sp = x.shape
        tokens, sp_after = self.patch_embed(x)
        if self.pos_embed is None:
            pe = torch.zeros(1, tokens.size(1), tokens.size(2), device=x.device, dtype=tokens.dtype)
            nn.init.trunc_normal_(pe, std=0.02)
        tokens = tokens + self.pos_embed

        # encode
        skips = self.vit(tokens)
        # reconstruct spatial maps
        maps = [s.transpose(1, 2).view(B, -1, *sp_after) for s in skips]

        # bottleneck
        out = maps[-1]
        seg_outs = []
        # decode
        for idx, (up, dec, head) in enumerate(zip(self.upconvs, self.decoders, self.seg_heads)):
            out = up(out)
            skip = maps[-(idx + 2)]
            if out.shape[2:] != skip.shape[2:]:
                mode = 'trilinear' if dim == 3 else 'bilinear'
                out = F.interpolate(out, size=skip.shape[2:], mode=mode, align_corners=False)
            out = torch.cat([skip, out], dim=1)
            out = dec(out)
            seg_outs.append(head(out))

        seg_outs = seg_outs[::-1]
        if self.deep_supervision:
            return seg_outs
        return seg_outs[0]