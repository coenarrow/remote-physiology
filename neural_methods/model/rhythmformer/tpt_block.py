"""The temporal periodic transformer stage of RhythmFormer."""

import math

import torch
from torch import nn

from neural_methods.model.rhythmformer.video_bi_former_block import VideoBiFormerBlock


class TPTBlock(nn.Module):
    """One temporal periodic transformer stage.

    Time is halved ``log2(t_patch)`` times by stride-2 convolutions, the
    routing-attention blocks run on the patched sequence, and the same number
    of 2x upsamples put the window length back.
    """

    def __init__(self, dim, depth, num_heads, t_patch, topk, region_split,
                 mlp_ratio=4., drop_path=0., side_dwconv=5):
        super().__init__()
        self.dim = dim
        self.depth = depth
        # --------- downsample layers & upsample layers ---------
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.layer_n = int(math.log(t_patch, 2))
        for i in range(self.layer_n):
            downsample_layer = nn.Sequential(
                nn.BatchNorm3d(dim),
                nn.Conv3d(dim, dim, kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            )
            self.downsample_layers.append(downsample_layer)
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=(2, 1, 1)),
                nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1, 0, 0)),
                nn.BatchNorm3d(dim),
                nn.ELU(),
            )
            self.upsample_layers.append(upsample_layer)
        # -------------------------------------------------------
        self.blocks = nn.ModuleList([
            VideoBiFormerBlock(
                dim=dim,
                region_split=region_split,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                num_heads=num_heads,
                t_patch=t_patch,
                topk=topk,
                mlp_ratio=mlp_ratio,
                side_dwconv=side_dwconv,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor):
        """``(N, C, D, H, W)`` -> ``(N, C, D, H, W)``."""
        for i in range(self.layer_n):
            x = self.downsample_layers[i](x)
        for blk in self.blocks:
            x = blk(x)
        for i in range(self.layer_n):
            x = self.upsample_layers[i](x)

        return x
