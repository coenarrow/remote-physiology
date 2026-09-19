"""The decoder unit of iBVPNet: a temporal transposed conv, then a spatial conv."""

import torch.nn as nn


class DeconvBlock3D(nn.Module):
    """``ConvTranspose3d`` in time and ``Conv3d`` in space, each followed by
    ``Tanh`` and ``InstanceNorm3d``."""

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super().__init__()
        k_t, k_s1, k_s2 = kernel_size
        s_t, s_s1, s_s2 = stride
        self.deconv_block_3d = nn.Sequential(
            nn.ConvTranspose3d(in_channel, in_channel, (k_t, 1, 1), (s_t, 1, 1), padding),
            nn.Tanh(),
            nn.InstanceNorm3d(in_channel),
            nn.Conv3d(in_channel, out_channel, (1, k_s1, k_s2), (1, s_s1, s_s2), padding),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.deconv_block_3d(x)
