"""The encoder of iBVPNet: a spatio-temporal stage, then a temporal one."""

import torch.nn as nn
import torch.nn.functional as F

from neural_methods.model._shared_modules.conv_block_3d import ConvBlock3D
from neural_methods.model._shared_modules.utils import nearest_multiple


class EncoderBlock(nn.Module):
    """Clip in, features at a quarter of the window length out."""

    def __init__(self, in_channel, filters, temporal_stride):
        """``filters`` are the five stages' filter counts and ``temporal_stride``
        the factor ``temporal_encoder`` takes time down by; both are iBVPNet's,
        passed in."""
        super().__init__()
        self.temporal_stride = temporal_stride

        # in_channel, out_channel, kernel_size, stride, padding
        self.spatio_temporal_encoder = nn.Sequential(
            ConvBlock3D(in_channel, filters[0], [1, 3, 3], [1, 1, 1], [0, 1, 1], bias=True),
            ConvBlock3D(filters[0], filters[1], [3, 3, 3], [1, 1, 1], [1, 1, 1], bias=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            ConvBlock3D(filters[1], filters[2], [1, 3, 3], [1, 1, 1], [0, 1, 1], bias=True),
            ConvBlock3D(filters[2], filters[3], [3, 3, 3], [1, 1, 1], [1, 1, 1], bias=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            ConvBlock3D(filters[3], filters[4], [1, 3, 3], [1, 1, 1], [0, 1, 1], bias=True),
            ConvBlock3D(filters[4], filters[4], [3, 3, 3], [1, 1, 1], [1, 1, 1], bias=True),
        )

        self.temporal_encoder = nn.Sequential(
            ConvBlock3D(filters[4], filters[4], [11, 1, 1], [1, 1, 1], [5, 0, 0], bias=True),
            ConvBlock3D(filters[4], filters[4], [11, 3, 3], [1, 1, 1], [5, 1, 1], bias=True),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
            ConvBlock3D(filters[4], filters[4], [11, 1, 1], [1, 1, 1], [5, 0, 0], bias=True),
            ConvBlock3D(filters[4], filters[4], [11, 3, 3], [1, 1, 1], [5, 1, 1], bias=True),
            nn.MaxPool3d((2, 2, 2), stride=(2, 1, 1)),
            ConvBlock3D(filters[4], filters[4], [7, 1, 1], [1, 1, 1], [3, 0, 0], bias=True),
            ConvBlock3D(filters[4], filters[4], [7, 3, 3], [1, 1, 1], [3, 1, 1], bias=True),
        )

    def forward(self, x):
        st_x = self.spatio_temporal_encoder(x)

        # Any window length: temporal_encoder halves time twice, so T must
        # divide by 4. At the paper's 160-frame windows the target equals T
        # and this is skipped.
        frames = st_x.shape[2]
        t4 = nearest_multiple(frames, self.temporal_stride)
        if t4 != frames:
            st_x = F.adaptive_avg_pool3d(st_x, (t4, st_x.shape[3], st_x.shape[4]))

        return self.temporal_encoder(st_x)
