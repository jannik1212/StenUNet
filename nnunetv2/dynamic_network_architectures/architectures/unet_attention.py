import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from typing import Union, Type, List, Tuple
from nnunetv2.dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int, conv_op, norm_op):
        super().__init__()
        self.W_g = nn.Sequential(
            conv_op(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            norm_op(F_int)
        )
        self.W_x = nn.Sequential(
            conv_op(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            norm_op(F_int)
        )
        self.psi = nn.Sequential(
            conv_op(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            norm_op(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ConvStack(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, conv_op, norm_op, norm_op_kwargs,
                 dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs):
        super().__init__()
        blocks = []
        for i in range(num_convs):
            in_ch = in_channels if i == 0 else out_channels
            blocks.append(conv_op(in_ch, out_channels, kernel_size=3, padding=1, bias=False))
            blocks.append(norm_op(out_channels, **norm_op_kwargs))
            blocks.append(nonlin(**nonlin_kwargs))
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class AttentionUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes,  # unused but needed for compatibility
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        super().__init__()
        self.deep_supervision = deep_supervision
        dim = convert_conv_op_to_dim(conv_op)

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        # Encoder
        self.encoders = nn.ModuleList()
        self.pooling = nn.ModuleList()
        in_channels = input_channels
        for s in range(n_stages):
            self.encoders.append(ConvStack(in_channels, features_per_stage[s], n_conv_per_stage[s],
                                           conv_op, norm_op, norm_op_kwargs,
                                           dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs))
            in_channels = features_per_stage[s]
            self.pooling.append(conv_op(in_channels, in_channels, kernel_size=2, stride=2))

        # Bottleneck
        self.bottleneck = ConvStack(features_per_stage[-1], features_per_stage[-1], n_conv_per_stage[-1],
                                    conv_op, norm_op, norm_op_kwargs,
                                    dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for d in range(n_stages - 2, -1, -1):
            self.upconvs.append(
                nn.ConvTranspose3d(features_per_stage[d + 1], features_per_stage[d], 2, 2)
                if conv_op == nn.Conv3d else
                nn.ConvTranspose2d(features_per_stage[d + 1], features_per_stage[d], 2, 2)
            )
            self.attentions.append(
                AttentionBlock(F_g=features_per_stage[d], F_l=features_per_stage[d],
                               F_int=features_per_stage[d] // 2,
                               conv_op=conv_op, norm_op=norm_op)
            )
            self.decoders.append(
                ConvStack(features_per_stage[d] * 2, features_per_stage[d], n_conv_per_stage_decoder[d],
                          conv_op, norm_op, norm_op_kwargs,
                          dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
            )

        self.final = conv_op(features_per_stage[0], num_classes, 1)

    def forward(self, x):
        enc_feats = []
        for enc, pool in zip(self.encoders, self.pooling):
            x = enc(x)
            enc_feats.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip = enc_feats[-(i + 2)]
            skip = self.attentions[i](x, skip)
            x = torch.cat((skip, x), dim=1)
            x = self.decoders[i](x)

        return self.final(x)