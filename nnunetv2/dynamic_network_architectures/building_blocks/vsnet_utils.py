"""
Utility functions for VSNet: window partitioning, reversing, and mask computation.
"""
import itertools
import torch
import torch.nn.functional as F
from typing import Sequence, Tuple, Optional


def window_partition(x: torch.Tensor, window_size: Sequence[int]) -> torch.Tensor:
    """
    Partition 4D/5D tensor into non-overlapping windows.
    x: (B, C, ...spatial dims...)
    window_size: tuple of length = spatial dims
    returns: (num_windows*B, window_volume, C)
    """
    dims = x.ndim - 2
    if dims == 2:
        B, C, H, W = x.shape
        ws_h, ws_w = window_size
        x = x.view(B, C, H // ws_h, ws_h, W // ws_w, ws_w)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.view(-1, ws_h * ws_w, C)
    elif dims == 3:
        B, C, D, H, W = x.shape
        ws_d, ws_h, ws_w = window_size
        x = x.view(B, C,
                   D // ws_d, ws_d,
                   H // ws_h, ws_h,
                   W // ws_w, ws_w)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        return x.view(-1, ws_d * ws_h * ws_w, C)
    else:
        raise ValueError(f"Unsupported input dims: {dims}")


def window_reverse(windows: torch.Tensor,
                   window_size: Sequence[int],
                   spatial_shape: Sequence[int]) -> torch.Tensor:
    """
    Reverse windows back to original spatial tensor.
    windows: (num_windows*B, window_volume, C)
    spatial_shape: tuple of spatial dims
    returns: (B, C, *spatial_shape)
    """
    dims = len(spatial_shape)
    if dims == 2:
        H, W = spatial_shape
        ws_h, ws_w = window_size
        num_windows_h = H // ws_h
        num_windows_w = W // ws_w
        B = windows.shape[0] // (num_windows_h * num_windows_w)
        x = windows.view(B, num_windows_h, num_windows_w, ws_h, ws_w, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(B, -1, H, W)
    elif dims == 3:
        D, H, W = spatial_shape
        ws_d, ws_h, ws_w = window_size
        num_windows_d = D // ws_d
        num_windows_h = H // ws_h
        num_windows_w = W // ws_w
        B = windows.shape[0] // (num_windows_d * num_windows_h * num_windows_w)
        x = windows.view(B,
                         num_windows_d, num_windows_h, num_windows_w,
                         ws_d, ws_h, ws_w,
                         -1)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        return x.view(B, -1, D, H, W)
    else:
        raise ValueError(f"Unsupported spatial dims: {dims}")


def get_window_size(
    input_shape: Sequence[int],
    window_size: Sequence[int],
    shift_size: Optional[Sequence[int]] = None
) -> Tuple[Sequence[int], Optional[Sequence[int]]]:
    """
    Adjust window and shift sizes so they are not larger than input dims.
    Returns (adjusted_window_size,) or (adjusted_window_size, adjusted_shift_size).
    """
    use_ws = list(window_size)
    use_ss = list(shift_size) if shift_size is not None else None
    for i, dim in enumerate(input_shape):
        if dim <= window_size[i]:
            use_ws[i] = dim
            if use_ss is not None:
                use_ss[i] = 0
    if shift_size is None:
        return tuple(use_ws), None
    return tuple(use_ws), tuple(use_ss)


def compute_mask(
    spatial_shape: Sequence[int],
    window_size: Sequence[int],
    shift_size: Sequence[int],
    device: torch.device
) -> torch.Tensor:
    """
    Compute attention mask for shifted windows.
    spatial_shape: (D, H, W) or (H, W)
    window_size, shift_size: same format
    returns: attn_mask of shape (num_windows, window_volume, window_volume)
    """
    dims = len(spatial_shape)
    if dims == 2:
        H, W = spatial_shape
        img_mask = torch.zeros((1, H, W, 1), device=device)
        cnt = 0
        for h_slice in (slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)):
            for w_slice in (slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)):
                img_mask[:, h_slice, w_slice, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask.permute(0,3,1,2), window_size)
        mask_windows = mask_windows.squeeze(-1)
    elif dims == 3:
        D, H, W = spatial_shape
        img_mask = torch.zeros((1, D, H, W, 1), device=device)
        cnt = 0
        for d_slice in (slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)):
            for h_slice in (slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)):
                for w_slice in (slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None)):
                    img_mask[:, d_slice, h_slice, w_slice, :] = cnt
                    cnt += 1
        # perm to (B,C,D,H,W) for partition
        mask_windows = window_partition(img_mask.permute(0,4,1,2,3), window_size)
        mask_windows = mask_windows.squeeze(-1)
    else:
        raise ValueError(f"Unsupported spatial dims: {dims}")
    # create attention mask
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask
