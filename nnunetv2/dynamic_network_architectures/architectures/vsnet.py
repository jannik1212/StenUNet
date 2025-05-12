# nnunetv2/dynamic_network_architectures/architectures/vsnet.py
"""
Wrapper for integrating VSNet into the nnU-Net dynamic network architecture API.
Follows the constructor pattern of other UNet variants for seamless use in get_network_from_plans.
Signature includes all nnU-Net args so it slots in dynamically; unused ones are noted.
"""
from typing import Union, Type, List, Tuple, Sequence
import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from nnunetv2.dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

# Core VSNet implementation should live under vsnet_core/vsnet.py
from nnunetv2.dynamic_network_architectures.architectures.vsnet_core.vsnet import VSNet as VSNetCore


class VSNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        # nnU-Net signature-only args (not used by VSNetCore unless forwarded)
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
        assert dim in (2, 3), "VSNetCore supports only 2D or 3D"
        self.deep_supervision = deep_supervision

        # dynamic config from nnU-Net plans
        depth = max(1, n_stages - 1)
        base_feature_size = (
            features_per_stage[0] if isinstance(features_per_stage, (list, tuple)) else features_per_stage
        )

        # optional forward of dropout to VSNetCore's drop_rate
        drop_rate = 0.0
        if dropout_op is not None and dropout_op_kwargs is not None:
            drop_rate = dropout_op_kwargs.get('p', dropout_op_kwargs.get('prob', 0.0))

        # instantiate core with dynamic and forwarded args
        self.net = VSNetCore(
            in_channels=input_channels,
            out_channels=num_classes,
            depth=depth,
            feature_size=base_feature_size,
            # pass through dropout rates
            drop_rate=drop_rate,
            attn_drop_rate=drop_rate,
            dropout_path_rate=drop_rate,
            # other VSNetCore args (img_size, num_heads, norm_name, etc.) use defaults
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through VSNetCore. Returns final logits or list of outputs for deep supervision.
        """
        outputs = self.net(x)
        if isinstance(outputs, (list, tuple)) and not self.deep_supervision:
            return outputs[-1]
        return outputs

    def compute_conv_feature_map_size(self, input_size: Tuple[int, ...]) -> int:
        """
        Approximate footprint by counting final segmentation output pixels/voxels.
        """
        return int(np.prod(input_size) * self.net.out_channels)
