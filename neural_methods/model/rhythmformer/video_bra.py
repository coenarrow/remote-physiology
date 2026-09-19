"""The bi-level routing attention of RhythmFormer.

Adapted from https://github.com/rayleizhu/BiFormer.
"""

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn import functional as F

from neural_methods.model._shared_modules.cdc_t import CDCT
from neural_methods.model.rhythmformer.utils import video_regional_routing_attention_torch


class VideoBRA(nn.Module):
    """Bi-level routing attention over a spatio-temporal token grid.

    The grid is cut into regions, each query region attends to the ``topk``
    key regions its pooled descriptor scores highest, and attention then runs
    token to token inside that selection. The region size is a forward-time
    quantity read off the grid, so no parameter here depends on the frame
    size or the window length. ``region_split`` is the number of regions per
    spatial axis, the model's ``REGION_SPLIT``.
    """

    def __init__(self, dim, region_split, num_heads=8, t_patch=8, qk_scale=None,
                 topk=4, side_dwconv=3):
        super().__init__()

        self.dim = dim
        self.region_split = region_split
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim must be divisible by num_heads!'
        self.head_dim = self.dim // self.num_heads
        self.scale = qk_scale or self.dim ** -0.5
        self.topk = topk
        self.t_patch = t_patch  # frame of patch
        # side_dwconv, i.e. LCE in the Shunted Transformer.
        self.lepe = nn.Conv3d(dim, dim, kernel_size=side_dwconv, stride=1,
                              padding=side_dwconv // 2, groups=dim) if side_dwconv > 0 else \
            (lambda x: torch.zeros_like(x))
        # Unused by the published forward, which projects q, k and v
        # separately below; kept as upstream has it, so the parameters are
        # created (and seeded) in the published order.
        self.qkv_linear = nn.Conv3d(self.dim, 3 * self.dim, kernel_size=1)
        self.output_linear = nn.Conv3d(self.dim, self.dim, kernel_size=1)
        self.proj_q = nn.Sequential(
            CDCT(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=0.2),
            nn.BatchNorm3d(dim),
        )
        self.proj_k = nn.Sequential(
            CDCT(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=0.2),
            nn.BatchNorm3d(dim),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
        )

    def forward(self, x: Tensor):
        frames, height, width = x.shape[2:]
        t_region = max(4 // self.t_patch, 1)
        region_size = (t_region, max(height // self.region_split, 1),
                       max(width // self.region_split, 1))

        # STEP 1: linear projection
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)

        # Any frame size: the regions have to tile the grid. Padding it up to
        # a whole number of them and cropping the attention output back is
        # empty at the paper's 8x8 token grid, which 2x2 regions tile exactly.
        # Replicate rather than zeros, so a padded region's pooled descriptor
        # stays a descriptor of real tokens.
        pad = tuple(-n % r for n, r in zip((frames, height, width), region_size))
        if any(pad):
            padding = (0, pad[2], 0, pad[1], 0, pad[0])
            q = F.pad(q, padding, mode="replicate")
            k = F.pad(k, padding, mode="replicate")
            v_pad = F.pad(v, padding, mode="replicate")
        else:
            v_pad = v

        # STEP 2: pre attention
        q_r = F.avg_pool3d(q.detach(), kernel_size=region_size, ceil_mode=True,
                           count_include_pad=False)
        k_r = F.avg_pool3d(k.detach(), kernel_size=region_size, ceil_mode=True,
                           count_include_pad=False)
        a_r = (rearrange(q_r, "n c t h w -> n (t h w) c")
               @ rearrange(k_r, "n c t h w -> n c (t h w)"))       # n(thw)(thw)
        # A short window on a small frame can leave fewer regions than the
        # paper routes; its deepest stage offers 320 and routes 40, so at the
        # paper's shape this clamp is not reached.
        idx_r = torch.topk(a_r, k=min(self.topk, a_r.shape[-1]), dim=-1).indices
        idx_r = repeat(idx_r, "n s k -> n m s k", m=self.num_heads)

        # STEP 3: refined attention
        output, _ = video_regional_routing_attention_torch(
            query=q, key=k, value=v_pad, scale=self.scale,
            region_graph=idx_r, region_size=region_size)
        if any(pad):
            output = output[:, :, :frames, :height, :width]

        output = output + self.lepe(v)      # nctHW
        return self.output_linear(output)   # nctHW
