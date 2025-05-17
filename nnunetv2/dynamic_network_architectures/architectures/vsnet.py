# nnunetv2/dynamic_network_architectures/architectures/vsnet.py
"""
Thin adapter to integrate the VSNetCore implementation into nnU-Net's dynamic architecture API.
Matches the constructor signature of other UNet variants so that get_network_from_plans can instantiate seamlessly.
nnU-Net will resample and crop volumes to size `kernel_sizes[0]` before forwarding, and this `img_size` is derived from it.
"""
from typing import Union, Type, List, Tuple, Sequence
import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from nnunetv2.dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

# Import the full VSNet implementation from vsnet_core
from nnunetv2.dynamic_network_architectures.architectures.vsnet_core.vsnet import VSNet as VSNetCore




class VSNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        # nnU-Net signature-only args (not forwarded)
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False
    ):
        super().__init__()
        dim = convert_conv_op_to_dim(conv_op)


        # derive core parameters
        raw_vs_feature_size = (
            features_per_stage[0]
            if isinstance(features_per_stage, (list, tuple))
            else features_per_stage
        )
        # adjust feature size so that hidden_size=16*fs divisible by 3 (VSNetCore uses num_heads=3)
        if raw_vs_feature_size % 3 != 0:
            vs_feature_size = ((raw_vs_feature_size + 2) // 3) * 3
        else:
            vs_feature_size = raw_vs_feature_size
        vs_depth = n_stages - 1

        # dropout settings
        drop_rate = dropout_op_kwargs.get('p', 0.0) if dropout_op_kwargs else 0.0
        attn_drop_rate = drop_rate
        dropout_path_rate = drop_rate

        img_size = 96

        # instantiate the core VSNet modele the core VSNet model
        self.net = VSNetCore(
            in_channels=input_channels,
            out_channels=num_classes,
            depth=vs_depth,
            img_size=img_size,
            num_heads=[3] * vs_depth if isinstance(vs_depth, int) else [3],
            feature_size=vs_feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            spatial_dims=dim
        )

        self.deep_supervision = False
        self.decoder = nn.Module()
        self.decoder.supervision = False

    def forward(self, x):
        """
        x: Tensor of shape (B, C, D, H, W)
        returns either
          - a single segmentation map Tensor (B, num_classes, D, H, W)
          - or a list of such maps if deep_supervision=True
        """
        out = self.net(x)
        if self.deep_supervision:
            return out
        return out[-1] if isinstance(out, (list, tuple)) else out

    def compute_conv_feature_map_size(self, input_size):
        # approximate by total voxels times sum of depths if available
        if hasattr(self.net, 'depths'):
            total = np.prod(input_size) * sum(self.net.depths)
        else:
            total = np.prod(input_size)
        return int(total)
