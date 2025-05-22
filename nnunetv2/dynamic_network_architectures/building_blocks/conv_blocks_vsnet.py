# dynamic_network_architectures/building_blocks/conv_blocks.py
"""
Convolutional and gating blocks for VSNet: DepTran, Gate, Outlayer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DepTran(nn.Module):
    """
    Depth-wise gated convolutional block.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=1, stride=1, padding=0, bias=True
        )
        self.groupconv1 = nn.Conv3d(
            in_channels, 2 * in_channels,
            kernel_size=1, stride=1, padding=0,
            bias=True, groups=in_channels
        )
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=True
        )
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.groupconv1(out)
        x1, x2 = torch.chunk(out, 2, dim=1)
        x1 = self.gelu(x1)
        x2 = self.sigmoid(x2)
        out = x1 * x2
        out = out + x
        out = self.conv2(out)
        out = self.relu(out)
        return out


class Gate(nn.Module):
    """
    Gating mechanism between encoder and decoder features.
    """
    def __init__(
        self,
        in_channels_up: int,
        in_channels_down: int,
        out_channels: int
    ):
        super().__init__()
        self.w1 = nn.Sequential(
            nn.Conv3d(
                in_channels_up, out_channels,
                kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.InstanceNorm3d(out_channels)
        )
        self.w2 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels_down, out_channels,
                kernel_size=2, stride=2, bias=False
            ),
            nn.Conv3d(
                out_channels, out_channels,
                kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.InstanceNorm3d(out_channels)
        )
        self.relu = nn.LeakyReLU(inplace=True)
        self.psi = nn.Sequential(
            nn.Conv3d(
                out_channels, 1,
                kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> torch.Tensor:
        w1 = self.w1(x1)
        w2 = self.w2(x2)
        psi = self.relu(w1 + w2)
        psi = self.psi(psi)
        return x1 * psi


class Outlayer(nn.Module):
    """
    Final output layer with activation.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels,
                kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.InstanceNorm3d(out_channels)
        )
        if activation == "Sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "Softmax":
            self.act = nn.Softmax(dim=1)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.act(x)
        return x
