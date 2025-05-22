import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from typing import Union, Type, List, Tuple
import numpy as np

from nnunetv2.dynamic_network_architectures.building_blocks.helper import (
    convert_conv_op_to_dim,
    get_matching_convtransp
)
from nnunetv2.dynamic_network_architectures.architectures.unet_attention import ConvStack
from nnunetv2.dynamic_network_architectures.building_blocks.swin_transformer import (
    SwinTransformerBlock,
    PatchEmbed,
    PatchMerging
)


class SwinUNet(nn.Module):
    """
    Dynamic Swin-UNet supporting 2D/3D segmentation in nnU-Net,
    with explicit kernel_size control for upsampling.
    """
    def __init__(
        self,
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
        dropout_op: Type[nn.Module] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.conv_op = conv_op
        self.num_classes = num_classes
        dim = convert_conv_op_to_dim(conv_op)
        assert dim in (2, 3), "SwinUNet only supports 2D or 3D."

        # normalize list args
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * (n_stages - 1)

        self.features_per_stage = list(features_per_stage)
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        base = self.features_per_stage[0]
        self.num_heads = [f // base for f in self.features_per_stage]
        self.window_size = 7

        # Patch embedding
        self.patch_embed = PatchEmbed(
            in_chans=input_channels,
            embed_dim=base,
            patch_size=strides[0],
            conv_op=conv_op,
            conv_bias=conv_bias,
            norm_layer=norm_op,
            norm_kwargs=norm_op_kwargs
        )

        # Encoder
        self.encoders = nn.ModuleList()
        self.pooling = nn.ModuleList()
        for s in range(n_stages):
            blocks = [
                SwinTransformerBlock(
                    dim=self.features_per_stage[s],
                    num_heads=self.num_heads[s],
                    window_size=self.window_size,
                    shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                    norm_layer=nn.LayerNorm
                ) for i in range(n_conv_per_stage[s])
            ]
            self.encoders.append(nn.Sequential(*blocks))
            if s < n_stages - 1:
                self.pooling.append(
                    PatchMerging(
                        in_dim=self.features_per_stage[s],
                        out_dim=self.features_per_stage[s + 1],
                        conv_op=conv_op,
                        norm_layer=norm_op,
                        norm_kwargs=norm_op_kwargs,
                        stride=strides[s + 1]
                    )
                )
            else:
                self.pooling.append(nn.Identity())

        # Bottleneck
        self.bottleneck = nn.Sequential(*[
            SwinTransformerBlock(
                dim=self.features_per_stage[-1],
                num_heads=self.num_heads[-1],
                window_size=self.window_size,
                shift_size=0,
                norm_layer=nn.LayerNorm
            ) for _ in range(n_conv_per_stage[-1])
        ])

        # Decoder
        transpconv_op = get_matching_convtransp(conv_op)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        for d in range(n_stages - 2, -1, -1):
            ks = self.kernel_sizes[d + 1]
            st = self.strides[d + 1]
            # upsample
            self.upconvs.append(
                transpconv_op(
                    self.features_per_stage[d + 1],
                    self.features_per_stage[d],
                    kernel_size=ks,
                    stride=st,
                    bias=conv_bias
                )
            )
            # conv stack
            self.decoders.append(
                ConvStack(
                    in_channels=self.features_per_stage[d] * 2,
                    out_channels=self.features_per_stage[d],
                    num_convs=n_conv_per_stage_decoder[d],
                    conv_op=conv_op,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs
                )
            )
            # segmentation head
            self.seg_layers.append(
                conv_op(
                    self.features_per_stage[d],
                    num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True
                )
            )

        # flag holder for deep supervision
        self.decoder = nn.Module()
        self.decoder.deep_supervision = self.deep_supervision

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = self.patch_embed(x)
        skips = []
        for enc, pool in zip(self.encoders, self.pooling):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)

        seg_outputs = []
        for i, (up, dec, seg) in enumerate(zip(self.upconvs, self.decoders, self.seg_layers)):
            x = up(x)
            skip = skips[-(i + 2)]
            if x.shape[2:] != skip.shape[2:]:
                mode = 'trilinear' if x.dim() == 5 else 'bilinear'
                x = F.interpolate(x, size=skip.shape[2:], mode=mode, align_corners=False)
            x = torch.cat((skip, x), dim=1)
            x = dec(x)
            seg_outputs.append(seg(x))

        seg_outputs = seg_outputs[::-1]
        use_ds = getattr(self.decoder, "deep_supervision", False)
        if use_ds:
            return seg_outputs
        return seg_outputs[0]

    def compute_conv_feature_map_size(self, input_size: Tuple[int, ...]) -> int:
        """
        Estimate memory footprint (#elements) of feature maps.
        """
        sizes = list(input_size)
        total = 0
        # encoder
        for s, pool in enumerate(self.pooling):
            total += self.features_per_stage[s] * int(np.prod(sizes))
            if not isinstance(pool, nn.Identity):
                for i in range(len(sizes)):
                    sizes[i] //= self.strides[s + 1]
        # bottleneck
        total += self.features_per_stage[-1] * int(np.prod(sizes))
        # decoder
        for d in range(len(self.upconvs)):
            for i in range(len(sizes)):
                sizes[i] *= self.strides[-(d + 1)]
            total += (self.features_per_stage[-(d + 2)] + self.num_classes) * int(np.prod(sizes))
        return int(total)
