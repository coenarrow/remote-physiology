"""The fast-to-slow lateral fusion of PhysMamba's two streams."""

import torch.nn as nn


class LateralConnection(nn.Module):
    """Stride the fast stream down to the slow stream's rate and add it in."""

    def __init__(self, fast_channels=32, slow_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(fast_channels, slow_channels, [3, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(slow_channels),
            nn.ReLU(),
        )

    def forward(self, slow_path, fast_path):
        fast_path = self.conv(fast_path)
        return fast_path + slow_path
