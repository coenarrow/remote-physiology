"""PhysFormer's feed-forward: a spatio-temporal conv between the two 1x1s."""

from einops import rearrange
from torch import nn


class PositionWiseFeedForwardST(nn.Module):
    """Feed-forward with a depth-wise spatio-temporal conv between the two 1x1s."""

    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Conv3d(dim, ff_dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        self.st_conv = nn.Sequential(
            nn.Conv3d(ff_dim, ff_dim, 3, stride=1, padding=1, groups=ff_dim, bias=False),
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv3d(ff_dim, dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(dim),
        )

    def forward(self, x, grid):
        gh, gw = grid
        x = rearrange(x, "b (gt gh gw) c -> b c gt gh gw", gh=gh, gw=gw)
        x = self.fc1(x)
        x = self.st_conv(x)
        x = self.fc2(x)
        return rearrange(x, "b c gt gh gw -> b (gt gh gw) c")
