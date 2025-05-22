# dynamic_network_architectures/building_blocks/attention_blocks.py
"""
Self-attention variants for VSNet: CSA (Channel Self-Attention), SSA (Spatial Self-Attention), and a generic SABlock.
"""
import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional


class CSA(nn.Module):
    """
    Channel Self-Attention: computes attention across channels for each spatial location.
    """
    def __init__(
        self,
        in_chans: int,
        img_size: int,
        dropout_rate: float = 0.0,
        save_attn: bool = False
    ):
        super().__init__()
        d = img_size // 16
        h = img_size // 16
        w = img_size // 16
        self.norm = nn.LayerNorm([d, h, w])
        self.scale = (d * h * w) ** -0.5
        self.groupconv = nn.Conv3d(
            in_channels=in_chans,
            out_channels=in_chans * 3,
            kernel_size=1,
            groups=in_chans
        )
        self.q_rearrange = rearrange
        self.k_rearrange = rearrange
        self.drop_weights = nn.Dropout(dropout_rate)
        self.drop_output = nn.Dropout(dropout_rate)
        self.save_attn = save_attn
        self.att_mat = torch.Tensor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        out = self.norm(x)
        out = self.groupconv(out)
        q, k, v = torch.chunk(out, 3, dim=1)
        # reshape: q: B, Cq/ , D,H,W -> B, Q, D*H*W
        B, C3, D, H, W = out.shape
        # channels split gives same C each
        C = C3 // 3
        q = rearrange(q, 'b c d h w -> b c (d h w)')
        k = rearrange(k, 'b c d h w -> b c (d h w)')
        attn = torch.einsum('bqc, bkc -> bqk', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        if self.save_attn:
            self.att_mat = attn.detach()
        attn = self.drop_weights(attn)
        v = rearrange(v, 'b c d h w -> b c (d h w)')
        out2 = torch.einsum('bqk, bkv -> bqv', attn, v)
        out2 = rearrange(out2, 'b c (d h w) -> b c d h w', d=D, h=H, w=W)
        out2 = self.drop_output(out2)
        return x + out2


class SSA(nn.Module):
    """
    Spatial Self-Attention: computes attention across spatial positions for each channel.
    """
    def __init__(
        self,
        hidden_size: int,
        img_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
        dim_head: Optional[int] = None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = hidden_size // num_heads if dim_head is None else dim_head
        self.inner_dim = self.dim_head * num_heads
        self.scale = self.dim_head ** -0.5
        d = img_size // 16
        h = img_size // 16
        w = img_size // 16
        self.input_rearrange = rearrange
        self.out_proj = nn.Linear(self.inner_dim, hidden_size)
        self.qkv = nn.Linear(hidden_size, self.inner_dim * 3, bias=qkv_bias)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.drop_output = nn.Dropout(dropout_rate)
        self.save_attn = save_attn
        self.att_mat = torch.Tensor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        # rearrange to (B, D*H*W, C)
        tokens = rearrange(x, 'b c d h w -> b (d h w) c')
        qkv = self.qkv(tokens)
        qkv = qkv.reshape(B, tokens.shape[1], 3, self.num_heads, self.dim_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, L, dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if self.save_attn:
            self.att_mat = attn.detach()
        attn = self.drop_weights(attn)
        out = attn @ v  # [B, heads, L, dim]
        out = out.permute(0, 2, 1, 3).reshape(B, tokens.shape[1], self.inner_dim)
        out = self.out_proj(out)
        out = self.drop_output(out)
        out = rearrange(out, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)
        return x + out


class SABlock(nn.Module):
    """
    Generic self-attention block (ViT-style) for feature maps.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
        dim_head: Optional[int] = None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = hidden_size // num_heads if dim_head is None else dim_head
        self.inner_dim = self.dim_head * num_heads
        self.scale = self.dim_head ** -0.5
        self.qkv = nn.Linear(hidden_size, self.inner_dim * 3, bias=qkv_bias)
        self.input_rearrange = rearrange
        self.out_proj = nn.Linear(self.inner_dim, hidden_size)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.drop_output = nn.Dropout(dropout_rate)
        self.save_attn = save_attn
        self.att_mat = torch.Tensor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) or (B, N, hidden_size)
        # flatten spatial dims if needed
        B, C, *sp = x.shape
        if len(sp) > 1:
            L = sp[0] * sp[1] if len(sp) == 2 else sp[0] * sp[1] * sp[2]
            tokens = rearrange(x, 'b c ... -> b (...) c')
        else:
            tokens = x
        B, L, _ = tokens.shape
        qkv = self.qkv(tokens).reshape(B, L, 3, self.num_heads, self.dim_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if self.save_attn:
            self.att_mat = attn.detach()
        attn = self.drop_weights(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, L, self.inner_dim)
        out = self.out_proj(out)
        out = self.drop_output(out)
        if len(sp) > 1:
            x_out = rearrange(out, 'b (d h w) c -> b c d h w', d=sp[0], h=sp[1], w=sp[2]) if len(sp) == 3 else rearrange(out, 'b (h w) c -> b c h w', h=sp[0], w=sp[1])
        else:
            x_out = out
        return x + x_out
