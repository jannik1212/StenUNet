# dynamic_network_architectures/building_blocks/swin_blocks.py
"""
Swin Transformer building blocks: WindowAttention, PatchMerging, SwinTransformerBlock, and SwinLayer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Optional, List, Tuple, Type, Union

from nnunetv2.dynamic_network_architectures.building_blocks.vsnet_utils import (
    window_partition,
    window_reverse,
    get_window_size,
    compute_mask
)


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention with relative position bias.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = tuple(window_size)
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # relative position bias table
        relative_size = 1
        for ws in self.window_size:
            relative_size *= (2 * ws - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(relative_size, num_heads)
        )
        coords = [torch.arange(ws) for ws in self.window_size]
        coords = torch.stack(torch.meshgrid(*coords, indexing='ij'))
        coords_flat = coords.flatten(1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        for i in range(len(self.window_size)):
            relative_coords[..., i] += self.window_size[i] - 1
        multipliers = [1]
        for ws in self.window_size[:-1][::-1]:
            multipliers.insert(0, multipliers[0] * (2 * ws - 1))
        index = relative_coords[..., 0] * multipliers[0]
        for i in range(1, len(self.window_size)):
            index += relative_coords[..., i] * multipliers[i]
        self.register_buffer('relative_index', index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        bias = self.relative_position_bias_table[self.relative_index.view(-1)].view(N, N, -1)
        bias = bias.permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(-1, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PatchMerging(nn.Module):
    def __init__(
        self,
        dim: int,
        norm_layer: Type[nn.LayerNorm] = nn.LayerNorm,
        spatial_dims: int = 3
    ):
        super().__init__()
        # factor = 2**spatial_dims (for 3D this is 8)
        factor = 2 ** spatial_dims
        self.reduction = nn.Linear(factor * dim, 2 * dim, bias=False)
        self.norm = norm_layer(factor * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        # first pad if any dimension is odd
        pd = (D % 2, H % 2, W % 2)
        if any(pd):
            pad = (0, pd[2], 0, pd[1], 0, pd[0])  # (W-l, W-r, H-l, H-r, D-l, D-r)
            x = F.pad(x, pad)
            B, C, D, H, W = x.shape

        # reshape into 2×2×2 blocks
        x = x.view(B, C, D//2, 2, H//2, 2, W//2, 2)
        # bring the block‐offset dims next to channel dim
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        # collapse the block dims into channel
        x = x.view(B, D//2, H//2, W//2, 8*C)

        # norm & linear reduce: (B, D', H', W', 8C) → (B, D', H', W', 2C)
        x = self.norm(x)
        x = self.reduction(x)

        # back to NCDHW
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


class SwinTransformerBlock(nn.Module):
    """
    Single Swin Transformer block with optional shift.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: Type[nn.LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.window_size = tuple(window_size)
        self.shift_size = tuple(shift_size)
        self.use_cp = use_checkpoint

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, num_heads, self.window_size, qkv_bias, attn_drop, drop)
        self.drop_path = nn.Identity()
        if drop_path > 0:
            from dynamic_network_architectures.building_blocks.vsnet_utils import DropPath
            self.drop_path = DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, *sp = x.shape     # original spatial dims
        x_in = x                # save un-padded input for the residual

        # 1) normalize and MLP prep on the _unpadded_ x_in, but we'll split this into
        #    two norms so both branches see a norm before their non-linearity.
        x = self.norm1(x.flatten(2).transpose(1, 2))
        x = x.transpose(1, 2).view(B, C, *sp)

        # 2) pad each spatial dim so it's divisible by the window size
        ws, _ = get_window_size(sp, self.window_size, None)
        pads = []
        for dim, w in zip(sp[::-1], ws[::-1]):
            rem = dim % w
            pads += [0, (w - rem) % w]
        x = F.pad(x, pads)
        sp = list(x.shape[2:])  # updated, padded spatial dims

        # 3) now do shifted windows attention exactly as before
        ws, ss = get_window_size(sp, self.window_size, self.shift_size)
        shifted = torch.roll(x, shifts=[-s for s in ss], dims=list(range(2, 2+len(sp)))) \
                  if any(ss) else x
        x_win = window_partition(shifted, ws)
        mask  = compute_mask(sp, ws, ss, x.device) if any(ss) else None
        attn  = self.attn(x_win, mask)
        x_    = window_reverse(attn, ws, sp)
        x_    = torch.roll(x_, shifts=ss, dims=list(range(2, 2+len(sp)))) \
                  if any(ss) else x_

        # 4) remove the padding so x_ is back to original [B,C,D,H,W]
        D0, H0, W0 = x_in.shape[2:]
        x_ = x_[..., :D0, :H0, :W0]

        # 5) residual + MLP residual
        x = x_in + self.drop_path(x_)
        
        # second norm+MLP on the UNPADDED spatial size:
        # x currently has shape [B,C,D0,H0,W0] (unpadded)
        D0, H0, W0 = x_in.shape[2:]             # original sizes
        # flatten spatial dims → (B, N0, C), N0 = D0*H0*W0
        y = self.norm2(x.flatten(2).transpose(1,2))
        # MLP on (B, N0, C)
        y = self.mlp(y)
        # reshape back to [B,C,D0,H0,W0]
        y = y.transpose(1,2).view(B, C, D0, H0, W0)
        x = x + self.drop_path(y)

        return x



class SwinLayer(nn.Module):
    """
    A stage of Swin blocks with optional downsample at end.
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: Union[Sequence[float], float] = 0.0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[nn.LayerNorm] = nn.LayerNorm,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        dpr = [drop_path] * depth if isinstance(drop_path, float) else drop_path
        for i in range(depth):
            shift = [0 if (i%2==0) else ws//2 for ws in window_size]
            self.blocks.append(
                SwinTransformerBlock(
                    dim, num_heads, window_size, shift,
                    mlp_ratio, qkv_bias, drop, attn_drop, dpr[i], norm_layer, use_checkpoint
                )
            )
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
