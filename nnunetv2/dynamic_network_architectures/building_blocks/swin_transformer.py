import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from typing import Type, Tuple, Union


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition feature map into non-overlapping windows for 2D or 3D tensors.
    x: (B, C, H, W) or (B, C, D, H, W)
    returns: (num_windows*B, window_size**dim, C)
    """
    dims = x.ndim - 2  # 2 for batch and channel
    if dims == 2:
        B, C, H, W = x.shape
        ws = window_size
        x = x.view(B, C, H // ws, ws, W // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, ws * ws, C)
        return x
    elif dims == 3:
        B, C, D, H, W = x.shape
        ws = window_size
        x = x.view(B, C,
                   D // ws, ws,
                   H // ws, ws,
                   W // ws, ws)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().view(-1, ws**3, C)
        return x
    else:
        raise ValueError(f"Unsupported input dims: {dims}")


def window_reverse(windows: torch.Tensor, window_size: int, spatial_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Reconstruct feature map from windows for 2D or 3D.
    windows: (num_windows*B, window_size**dim, C)
    spatial_shape: (H, W) or (D, H, W)
    returns: (B, C, *spatial_shape)
    """
    dim = len(spatial_shape)
    ws = window_size
    if dim == 2:
        H, W = spatial_shape
        B = int(windows.shape[0] // (H * W / ws / ws))
        x = windows.view(B, H // ws, W // ws, ws, ws, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
        return x
    elif dim == 3:
        D, H, W = spatial_shape
        B = int(windows.shape[0] // (D * H * W / ws**3))
        x = windows.view(B,
                         D // ws, H // ws, W // ws,
                         ws, ws, ws,
                         -1)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous().view(B, -1, D, H, W)
        return x
    else:
        raise ValueError(f"Unsupported spatial dims: {dim}")


class Mlp(nn.Module):
    """
    Feed-forward network.
    """
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, dropout: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module.
    Supports 2D and 3D windows.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size: int,
                 qkv_bias: bool = True,
                 dropout: float = 0.,
                 attn_dropout: float = 0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Basic Swin Transformer block supporting 2D or 3D inputs.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size: int = 7,
                 shift_size: int = 0,
                 mlp_ratio: float = 4.,
                 dropout: float = 0.,
                 attn_dropout: float = 0.,
                 norm_layer: Type[nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, num_heads, window_size,
                                    qkv_bias=True, dropout=dropout, attn_dropout=attn_dropout)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = x.ndim - 2
        shape = x.shape[2:]
        shortcut = x
        # LayerNorm
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.norm1(x_flat)
        x = x_flat.transpose(1, 2).view(x.shape)

        # padding
        pads = []
        for s, size in zip(reversed(shape), [self.window_size]*dims):
            pad = (size - s % size) % size
            pads.extend([0, pad])
        # F.pad expects reverse order: last two dims first
        x = F.pad(x, pads)
        padded_shape = x.shape[2:]

        # shift
        if self.shift_size > 0:
            shifts = [-self.shift_size] * dims
            dims_idx = list(range(2, 2 + dims))
            x = torch.roll(x, shifts=shifts, dims=dims_idx)

        # partition
        x_windows = window_partition(x, self.window_size)
        attn_windows = self.attn(x_windows)

        # reverse windows
        x = window_reverse(attn_windows, self.window_size, padded_shape)

        # reverse shift
        if self.shift_size > 0:
            shifts = [self.shift_size] * dims
            dims_idx = list(range(2, 2 + dims))
            x = torch.roll(x, shifts=shifts, dims=dims_idx)

        # remove padding
        slices = [slice(0, s) for s in shape]
        x = x[(..., *slices)]
        x = shortcut + x

        # MLP
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.norm2(x_flat)
        x2 = self.mlp(x_flat)
        x2 = x2.transpose(1, 2).view(x.shape)
        x = x + x2
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding for 2D or 3D via conv.
    """
    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        patch_size: Union[int, Tuple[int, ...]] = 4,
        conv_op: Type[_ConvNd] = nn.Conv2d,
        conv_bias: bool = False,
        norm_layer: Type[nn.Module] = None,
        norm_kwargs: dict = None
    ):
        super().__init__()
        if isinstance(patch_size, int):
            kernel_size = patch_size
            stride = patch_size
        else:
            kernel_size = patch_size
            stride = patch_size
        self.proj = conv_op(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=conv_bias
        )
        if norm_layer is not None:
            # e.g. nn.InstanceNorm2d or nn.InstanceNorm3d
            self.norm = norm_layer(embed_dim, **(norm_kwargs or {}))
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, C, H, W] (2D) or [B, C, D, H, W] (3D)
        x = self.proj(x)
        if self.norm is not None:
            # Apply InstanceNorm2d/3d directly on the conv output
            x = self.norm(x)
        return x



class PatchMerging(nn.Module):
    """
    Patch Merging for downsampling 2D or 3D via conv.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        conv_op: Type[_ConvNd] = nn.Conv2d,
        norm_layer: Type[nn.Module] = None,
        norm_kwargs: dict = None,
        stride: Union[int, Tuple[int, ...]] = 2
    ):
        super().__init__()
        self.reduction = conv_op(
            in_dim,
            out_dim,
            kernel_size=stride,
            stride=stride,
            bias=False
        )
        if norm_layer is not None:
            # e.g. nn.InstanceNorm2d or nn.InstanceNorm3d
            self.norm = norm_layer(out_dim, **(norm_kwargs or {}))
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, C, H, W] or [B, C, D, H, W]
        x = self.reduction(x)
        if self.norm is not None:
            # Apply InstanceNorm2d/3d directly
            x = self.norm(x)
        return x

