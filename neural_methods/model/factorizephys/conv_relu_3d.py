"""The 1x1x1 convolution block of FSAM's post-factorisation stage."""

import torch.nn as nn


class ConvReLU3D(nn.Module):
    """A 1x1x1 convolution and a ReLU.

    Upstream's ``ConvBNReLU`` with the branches FactorizePhys never took
    removed: the instance norm was never switched on and the 1-D / 2-D
    convolutions were never built.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1),
                              stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))
