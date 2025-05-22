# dynamic_network_architectures/architectures/vsnet.py
"""
VSNet integration for nnU-Net pipeline.
Matches nnUNet dynamic-architecture constructor signature.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Type
from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.attention_blocks_vsnet import CSA, SSA
from dynamic_network_architectures.building_blocks.conv_blocks_vsnet import DepTran, Gate
from dynamic_network_architectures.building_blocks.swin_blocks_vsnet import SwinLayer, PatchMerging
from nnunetv2.dynamic_network_architectures.building_blocks.helper import (
    convert_conv_op_to_dim,
    get_matching_convtransp
)
from dynamic_network_architectures.building_blocks.unetr_blocks import ConvStackND


class VSNet(nn.Module):
    """
    Wrapper so Trainer sees the expected API.
    """
    def __init__(
        self,
        input_channels:  int,
        n_stages:         int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op:          Type[_ConvNd],
        kernel_sizes:     Union[int, List[int], Tuple[int, ...]],
        strides:          Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes:      int,
        n_conv_per_stage_decoder: Union[int, List[int], Tuple[int, ...]],
        conv_bias:        bool = False,
        norm_op:          Type[nn.Module] = nn.InstanceNorm3d,
        norm_op_kwargs:   dict = None,
        dropout_op:       Type[nn.Module] = None,
        dropout_op_kwargs:dict = None,
        nonlin:           Type[nn.Module] = nn.GELU,
        nonlin_kwargs:    dict = None,
        deep_supervision: bool = False
    ):
        super().__init__()
        # instantiate the core network
        self.net = VSNetCore(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision
        )
        # pipeline attributes
        self.encoders    = self.net.encoders
        self.pooling     = self.net.pooling
        self.bottleneck  = self.net.swin
        self.upconvs     = self.net.upconvs
        self.decoders    = self.net.decoders
        self.seg_layers  = self.net.seg_heads
        self.decoder     = nn.Module()
        self.decoder.deep_supervision = deep_supervision

    def forward(self, x: torch.Tensor):
        return self.net(x)


class VSNetCore(nn.Module):
    """
    VSNet main network, wired to nnU-Net dynamic pipeline signature.
    Supports 3D volumes with 4 encoder/decoder stages.
    """
    def __init__(
        self,
        input_channels:  int,
        n_stages:         int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op:          Type[_ConvNd],
        kernel_sizes:     Union[int, List[int], Tuple[int, ...]],
        strides:          Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes:      int,
        n_conv_per_stage_decoder: Union[int, List[int], Tuple[int, ...]],
        conv_bias:        bool = False,
        norm_op:          Type[nn.Module] = nn.InstanceNorm3d,
        norm_op_kwargs:   dict = None,
        dropout_op:       Type[nn.Module] = None,
        dropout_op_kwargs:dict = None,
        nonlin:           Type[nn.Module] = nn.GELU,
        nonlin_kwargs:    dict = None,
        deep_supervision: bool = False
    ):
        super().__init__()
        assert convert_conv_op_to_dim(conv_op) == 3, "VSNetCore supports only 3D"
        # must have 4 stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        assert len(features_per_stage) == 4, "features_per_stage must be length 4"

        def _to_list(x, length):
            return x if isinstance(x, (list, tuple)) else [x] * length

        features = list(features_per_stage)
        strides = _to_list(strides, 4)
        kernel_sizes = _to_list(kernel_sizes, 4)
        convs = _to_list(n_conv_per_stage, 4)
        dec_convs = _to_list(n_conv_per_stage_decoder, 4)

        # --- Encoder ---
        self.encoders = nn.ModuleList()
        for i in range(4):
            in_ch, out_ch = (input_channels, features[0]) if i == 0 else (features[i-1], features[i])
            # use ConvStackND for encoder
            self.encoders.append(
                ConvStackND(
                    in_ch, out_ch, convs[i],
                    conv_op=conv_op,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs
                )
            )
        # pooling
        self.pooling = nn.ModuleList([
            nn.Identity(),
            nn.MaxPool3d(strides[1]),
            nn.MaxPool3d(strides[2]),
            nn.MaxPool3d(strides[3])
        ])
        # gating
        self.gates = nn.ModuleList([
            Gate(features[i-1], features[i], features[i-1])
            for i in range(1,4)
        ])

        # --- Bottleneck: Swin ---
        window_size = kernel_sizes[3]
        self.swin = SwinLayer(
            dim=features[-1], depth=convs[-1], num_heads=features[-1]//features[0],
            window_size=(7,7,7), downsample=PatchMerging
        )
        self.CSA = CSA(in_chans=features[-1], img_size=kernel_sizes[0])
        self.SSA = SSA(hidden_size=features[-1], img_size=kernel_sizes[0], num_heads=features[-1]//features[0])

        # depth-transfer
        self.dt = nn.ModuleList([DepTran(f, f) for f in 
                                  [features[3], features[2], features[1], features[0]]])

        # --- Decoder ---
        transp = get_matching_convtransp(conv_op)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.seg_heads = nn.ModuleList()
        for i in range(4):
            in_ch = features[3-i]
            out_ch = features[2-i] if i < 3 else features[0]
            ks = kernel_sizes[3-i]
            st = strides[3-i]
            self.upconvs.append(transp(in_ch, out_ch, ks, st, bias=conv_bias))
            # decoder conv-stack
            self.decoders.append(
                ConvStackND(
                    out_ch*2, out_ch, dec_convs[3-i],
                    conv_op=conv_op,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs
                )
            )
            self.seg_heads.append(nn.Conv3d(out_ch, num_classes, 1, bias=True))

        self.decoder = nn.Module()
        self.decoder.deep_supervision = deep_supervision

    def forward(self, x: torch.Tensor):
        # encoder
        x1 = self.encoders[0](x)
        x2 = self.encoders[1](x1); x2p = self.pooling[1](x2)
        x1g = self.gates[0](x1, x2p)
        x3 = self.encoders[2](x2p); x3p = self.pooling[2](x3)
        x2g = self.gates[1](x2p, x3p)
        x4 = self.encoders[3](x3p); x4p = self.pooling[3](x4)
        x3g = self.gates[2](x3p, x4p)
        # bottleneck
        x5 = self.swin(x4p); x5 = self.CSA(x5); x5 = self.SSA(x5)
        # decoder
        skips = [x4, x3g, x2g, x1g]
        outs = []
        x_cur = x5
        for i in range(4):
            x_cur = self.dt[i](x_cur)
            x_cur = self.upconvs[i](x_cur)
            skip = skips[i]
            if x_cur.shape[2:] != skip.shape[2:]:
                x_cur = F.interpolate(x_cur, skip.shape[2:], 'trilinear', False)
            x_cur = torch.cat([skip, x_cur], 1)
            x_cur = self.decoders[i](x_cur)
            outs.append(self.seg_heads[i](x_cur))
        return outs if self.decoder.deep_supervision else outs[0]