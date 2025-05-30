"""
Self-attention variants for VSNet: CSA (Channel Self-Attention), SSA (Spatial Self-Attention), and a generic SABlock.
"""
import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional

class CSA(nn.Module):
    """
    Channel Self-Attention with extra numerical guards.
    """
    def __init__(self, in_chans: int, dropout_rate: float = 0.0, save_attn: bool = False):
        super().__init__()
        self.norm         = nn.LayerNorm(in_chans)
        self.groupconv    = nn.Conv3d(in_chans, in_chans*3, kernel_size=1, groups=in_chans)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.drop_output  = nn.Dropout(dropout_rate)
        self.save_attn    = save_attn
        self.att_mat      = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape

        # 1) LayerNorm & project to Q,K,V
        y = x.permute(0,2,3,4,1)                  # (B, D, H, W, C)
        y = self.norm(y).permute(0,4,1,2,3)       # (B, C, D, H, W)
        y = self.groupconv(y)                     # (B, 3C, D, H, W)
        q, k, v = torch.chunk(y, 3, dim=1)        # each (B, C, D, H, W)

        # 2) flatten spatial dims
        q = rearrange(q, 'b c d h w -> b c (d h w)')
        k = rearrange(k, 'b c d h w -> b c (d h w)')
        v = rearrange(v, 'b c d h w -> b c (d h w)')

        # 3) channel-wise logits scaled by √C
        logits = torch.einsum('bcn,bkn->bck', q, k) * (C ** -0.5)

        # 4) stabilize: subtract max, clamp, NaN→0
        logits = logits - logits.amax(-1, keepdim=True)

        # 5) softmax & dropout
        attn = logits.softmax(-1)
        if self.save_attn:
            self.att_mat = attn.detach()
        attn = self.drop_weights(attn)

        # 6) apply to V, reshape back
        out = torch.einsum('bck,bkn->bcn', attn, v)
        out = out.view(B, C, D, H, W)
        out = self.drop_output(out)

        # 7) residual
        return x + out


class SSA(nn.Module):
    """
    Spatial Self-Attention with extra numerical guards.
    """
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 dropout_rate: float = 0.0,
                 qkv_bias: bool = False,
                 save_attn: bool = False,
                 dim_head: int = None):
        super().__init__()
        self.num_heads    = num_heads
        self.dim_head     = (hidden_size // num_heads) if dim_head is None else dim_head
        self.inner_dim    = self.dim_head * num_heads
        self.scale        = self.dim_head ** -0.5

        self.qkv          = nn.Linear(hidden_size, self.inner_dim*3, bias=qkv_bias)
        self.out_proj     = nn.Linear(self.inner_dim, hidden_size)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.drop_output  = nn.Dropout(dropout_rate)
        self.save_attn    = save_attn
        self.att_mat      = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        N = D * H * W

        # 1) flatten spatial → tokens
        tokens = rearrange(x, 'b c d h w -> b (d h w) c')  # (B, N, C)

        # 2) project to QKV and reshape
        qkv = self.qkv(tokens)                             # (B, N, 3*inner_dim)
        qkv = qkv.view(B, N, 3, self.num_heads, self.dim_head)
        qkv = qkv.permute(2,0,3,1,4)                       # (3, B, heads, N, dim_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2a) clamp Q/K
        q = q.clamp(-5.0, 5.0)
        k = k.clamp(-5.0, 5.0)

        # 3) raw attention logits scaled
        logits = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, N, N)

        # 4) stabilize
        logits = logits - logits.amax(-1, keepdim=True)

        # 5) softmax & dropout
        attn = logits.softmax(-1)
        if self.save_attn:
            self.att_mat = attn.detach()
        attn = self.drop_weights(attn)

        # 6) attend & project
        out = attn @ v                                     # (B, heads, N, dim_head)
        out = out.permute(0,2,1,3).reshape(B, N, self.inner_dim)
        out = self.out_proj(out)
        out = self.drop_output(out)

        # 7) reshape + residual
        out = rearrange(out, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)
        return x + out



class SABlock(nn.Module):
    """
    Generic self-attention block (ViT-style) for 2D or 3D feature maps.
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
        self.dim_head  = hidden_size // num_heads if dim_head is None else dim_head
        self.inner_dim = self.dim_head * num_heads
        self.scale     = self.dim_head ** -0.5

        self.qkv          = nn.Linear(hidden_size, self.inner_dim * 3, bias=qkv_bias)
        self.out_proj     = nn.Linear(self.inner_dim, hidden_size)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.drop_output  = nn.Dropout(dropout_rate)
        self.save_attn    = save_attn
        self.att_mat      = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # support x: (B, C, H, W), (B, C, D, H, W) or (B, N, hidden_size)
        B, C, *S = x.shape

        # flatten spatial dims if present
        if len(S) > 1:
            L = int(torch.prod(torch.tensor(S)))
            tokens = rearrange(x, 'b c ... -> b (...) c')
        else:
            tokens = x
            L = tokens.shape[1]

        # Q/K/V
        qkv = self.qkv(tokens).view(B, L, 3, self.num_heads, self.dim_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)                  # (3, B, heads, L, dim_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # attention
        attn = (q @ k.transpose(-2, -1)) * self.scale    # (B, heads, L, L)
        attn = attn.softmax(dim=-1)
        if self.save_attn:
            self.att_mat = attn.detach()
        attn = self.drop_weights(attn)

        # apply to V
        out = (attn @ v).transpose(1, 2).reshape(B, L, self.inner_dim)
        out = self.out_proj(out)
        out = self.drop_output(out)

        # reshape back if needed
        if len(S) > 1:
            if len(S) == 3:
                out = rearrange(out, 'b (d h w) c -> b c d h w', d=S[0], h=S[1], w=S[2])
            else:
                out = rearrange(out, 'b (h w) c -> b c h w', h=S[0], w=S[1])
        return x + out
