"""The Mamba stage of a PhysMamba temporal-difference block.

All reshaping is einops.
"""

import math

import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import DropPath, trunc_normal_

from neural_methods.model.physmamba.mamba_compat import make_mamba


class MambaLayer(nn.Module):
    """A bidirectional Mamba over every spatio-temporal position of a clip.

    ``(B, dim, T, H, W)`` in, the same shape out, always in float32.

    ``mamba`` is the sequence module, ``(B, L, dim) -> (B, L, dim)``; left
    out, it is the published bidirectional Mamba1. PhysMamba2 passes its own.
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, channel_token=False, mamba=None):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        drop_path = 0
        self.mamba = mamba if mamba is not None else make_mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba=True,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_patch_token(self, x):
        """Every spatio-temporal position is one token of the SSM sequence."""
        assert x.shape[1] == self.dim, f"expected {self.dim} channels, got {x.shape[1]}"
        frames, height, width = x.shape[2:]
        x_flat = rearrange(x, "b d t h w -> b (t h w) d")
        x_norm = self.norm1(x_flat)
        x_mamba = self.mamba(x_norm)
        x_out = self.norm2(x_flat + self.drop_path(x_mamba))
        return rearrange(x_out, "b (t h w) d -> b d t h w", t=frames, h=height, w=width)

    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
        out = self.forward_patch_token(x)
        return out
