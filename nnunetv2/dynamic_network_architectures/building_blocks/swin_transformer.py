import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from typing import Type, Tuple, Union

class RelativePositionBias(nn.Module):
    """
    Implements relative position bias for Swin Transformer.
    Works for both 2D and 3D based on the given window size.
    """
    def __init__(self, window_size, num_heads, is_3d=False):
        super().__init__()
        self.is_3d = is_3d
        self.window_size = window_size if isinstance(window_size, (tuple, list)) else (window_size,) * (3 if is_3d else 2)
        self.num_heads = num_heads

        relative_position_dims = [(2 * s - 1) for s in self.window_size]
        num_relative_positions = int(torch.prod(torch.tensor(relative_position_dims)))

        # learnable bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_positions, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # create a relative position index for each token in the window
        coords = [torch.arange(s) for s in self.window_size]
        coords = torch.stack(torch.meshgrid(*coords, indexing="ij"))  # dims x window_volume
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (dims x N x N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N x N x dims)

        for i in range(len(self.window_size)):
            relative_coords[:, :, i] += self.window_size[i] - 1

        if self.is_3d:
            idx = (relative_coords[:, :, 0] * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1) +
                   relative_coords[:, :, 1] * (2 * self.window_size[2] - 1) +
                   relative_coords[:, :, 2])
        else:
            idx = (relative_coords[:, :, 0] * (2 * self.window_size[1] - 1) +
                   relative_coords[:, :, 1])

        self.register_buffer("relative_position_index", idx)

    def forward(self):
        """
        Returns: bias tensor of shape (num_heads, N, N)
        """
        relative_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_bias = relative_bias.view(
            self.relative_position_index.shape[0],
            self.relative_position_index.shape[1],
            -1  # num_heads
        )  # shape (N, N, num_heads)
        return relative_bias.permute(2, 0, 1)  # (num_heads, N, N)

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
        # compute batch size by integer arithmetic
        num_windows_per_image = (H // ws) * (W // ws)
        B = windows.shape[0] // num_windows_per_image

        x = windows.view(B, H // ws, W // ws, ws, ws, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
        return x

    elif dim == 3:
        D, H, W = spatial_shape
        # compute batch size by integer arithmetic
        num_windows_per_image = (D // ws) * (H // ws) * (W // ws)
        B = windows.shape[0] // num_windows_per_image

        x = windows.view(
            B,
            D // ws, H // ws, W // ws,
            ws,      ws,      ws,
            -1
        )
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
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size: Union[int, Tuple[int, ...]],
                 qkv_bias: bool = True,
                 dropout: float = 0.,
                 attn_dropout: float = 0.,
                 is_3d: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.is_3d = is_3d

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        self.relative_position_bias = RelativePositionBias(
            window_size=window_size,
            num_heads=num_heads,
            is_3d=is_3d
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.relative_position_bias()
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
        self.attn = WindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=True,
            dropout=dropout,
            attn_dropout=attn_dropout,
            is_3d=True if dim == 3 else False
        )
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

        # padding so H, W (and D) are multiples of window_size
        if dims == 2:
            _, _, H, W = x.shape
            pad_h = (self.window_size - H % self.window_size) % self.window_size
            pad_w = (self.window_size - W % self.window_size) % self.window_size
            # pad = (left, right, top, bottom)
            x = F.pad(x, (0, pad_w, 0, pad_h))
        elif dims == 3:
            _, _, D, H, W = x.shape
            pad_d = (self.window_size - D % self.window_size) % self.window_size
            pad_h = (self.window_size - H % self.window_size) % self.window_size
            pad_w = (self.window_size - W % self.window_size) % self.window_size
            # pad = (left, right, top, bottom, front, back)
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        else:
            raise ValueError(f"Unsupported input dims: {dims}")

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

