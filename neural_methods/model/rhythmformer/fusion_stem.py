"""The two-path fusion stem of RhythmFormer."""

import torch
from einops import rearrange
from torch import nn


class FusionStem(nn.Module):
    """Two-path stem: the frame itself and its four temporal differences.

    ``in_channels`` is the only width that differs from the published network;
    at ``3`` the difference path takes the original's 12 planes, four frame
    differences of three channels.
    """

    def __init__(self, in_channels: int = 3, dim: int = 64, alpha=0.5, beta=0.5):
        super().__init__()

        self.stem11 = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )

        self.stem12 = nn.Sequential(
            nn.Conv2d(4 * in_channels, dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )

        self.stem21 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        self.stem22 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        """``(N, D, C, H, W)`` -> ``(N*D, dim, H/4, W/4)``."""
        frames = x.shape[1]
        x1 = torch.cat([x[:, :1], x[:, :1], x[:, :frames - 2]], 1)
        x2 = torch.cat([x[:, :1], x[:, :frames - 1]], 1)
        x3 = x
        x4 = torch.cat([x[:, 1:], x[:, frames - 1:]], 1)
        x5 = torch.cat([x[:, 2:], x[:, frames - 1:], x[:, frames - 1:]], 1)
        diff = torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4], 2)
        x_diff = self.stem12(rearrange(diff, "n d c h w -> (n d) c h w"))
        x = self.stem11(rearrange(x3, "n d c h w -> (n d) c h w"))

        # fusion layer 1
        x_path1 = self.alpha * x + self.beta * x_diff
        x_path1 = self.stem21(x_path1)
        # fusion layer 2
        x_path2 = self.stem22(x_diff)
        return self.alpha * x_path1 + self.beta * x_path2
