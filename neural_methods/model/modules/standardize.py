import torch
import torch.nn as nn
from einops import reduce


class Standardize(nn.Module):
    """Z-score each (sample, channel) block with its own mean and std.

    Input and output are ``(B, C, T, H, W)``. Statistics are taken over
    ``t h w`` so a channel never depends on the others.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = reduce(x, "b c t h w -> b c 1 1 1", "mean")
        std = reduce(x, "b c t h w -> b c 1 1 1", torch.std)
        return torch.nan_to_num((x - mean) / std.clamp_min(self.eps))